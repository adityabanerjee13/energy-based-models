import random
import numpy as np
import torch
from torch import nn
from utils import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Trainer():
    def __init__(
        self,
        model,
        trained=False,
        learning_rate = 1e-3,
        clipnorm = 100.,
        n_slices = 1,
        loss_type = 'ssm-vr',
        noise_type = 'gaussian',
        percent_model_sample = 0.9,
        batch_size = 100,
        max_len = 500,
        alpha = 1
    ):
        """Energy based model trainer

        Args:
            model (nn.Module): energy-based model
            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            clipnorm (float, optional): gradient clip. Defaults to 100..
            n_slices (int, optional): number of slices for sliced score matching loss.
                Defaults to 1.
            loss_type (str, optional): type of loss. Can be 'ssm-vr', 'ssm', 'deen',
                'dsm'. Defaults to 'ssm-vr'.
            noise_type (str, optional): type of noise. Can be 'radermacher', 'sphere'
                or 'gaussian'. Defaults to 'radermacher'.
            device (str, optional): torch device. Defaults to 'cuda'.
        """
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.loss_type = loss_type.lower()
        self.clipnorm = clipnorm
        if self.loss_type == 'ssm_vr':
            self.noise_type = noise_type.lower()
            self.n_slices = n_slices
        elif self.loss_type == 'ebm':
            self.alpha = alpha
            self.shape = (2,)
            self.sample_size = batch_size
            self.percent_model_sample = percent_model_sample
            self.max_len = max_len
            self.examples = [(torch.rand((1,) + self.shape) * 6 - 3) for _ in range(self.sample_size)]
            # print(len(self.examples))

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.num_gradsteps = 0
        self.num_epochs = 0
        self.progress = 0
        self.tb_writer = None

    def ssm_vr_loss(self, x, v):
        """SSM-VR (variance reduction) loss from
        Sliced Score Matching: A Scalable Approach to Density and Score Estimation

        The loss is computed as
        s = -dE(x)/dx
        loss = vT*(ds/dx)*v + 1/2*||s||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises

        Returns:
            SSM-VR loss
        """
        x = x.unsqueeze(0).expand(self.n_slices, *x.shape) # (n_slices, b, ...)
        x = x.contiguous().view(-1, *x.shape[2:]) # (n_slices*b, ...)
        x = x.requires_grad_()
        score = self.model.score(x) # (n_slices*b, ...)
        sv = torch.sum(score * v) # ()
        loss1 = torch.norm(score, dim=-1) ** 2 * 0.5 # (n_slices*b,)
        gsv = torch.autograd.grad(sv, x, create_graph=True)[0] # (n_slices*b, ...)
        loss2 = torch.sum(v*gsv, dim=-1) # (n_slices*b,)
        loss = (loss1 + loss2).mean() # ()

        return loss
    
    def ebm_loss(self, x):
        """EBM (variance reduction) loss from
        Energy based models: modeling E: X -> R as a potential function

        The loss is computed as
        E_real(x) = -model(x_real)
        E_aug(x) = -model(x_aug)
        loss = E_aug - E_real + alpha.(E_real^2 + E_aug^2)

        Args:
            x (torch.Tensor): input samples

        Returns:
            SSM-VR loss
        """
        def sample_new_exmps_buffer(steps,step_size):
            n_new = np.random.binomial(self.sample_size, 1 - self.percent_model_sample)
            rand_samples = torch.rand((n_new,) + self.shape) * 2 - 1
            # print(self.sample_size, n_new, len(self.examples))
            old_samples = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
            inp_samples = torch.cat([rand_samples, old_samples], dim=0).detach().to(device)
            score_fn = lambda x: self.model.score(x)
            self.model.eval()
            out = langevin_dynamics(score_fn, inp_samples, step_size, steps)

            self.examples = list(out.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
            self.examples = self.examples[:self.max_len]
            return out

        # Real samples
        real_samples = x.clone().detach()
        
        # Obtain samples
        aug_samples = sample_new_exmps_buffer(steps=60, step_size=0.01)

        # Predict energy score for all images
        inp_samples = torch.cat([real_samples, aug_samples], dim=0).to(device)
        # print(self.model.forward(inp_samples).chunk(2, dim=0))
        self.model.train()
        real_out, aug_out = (-self.model.forward(inp_samples)).chunk(2, dim=0)
        
        # Calculate losses
        reg_loss = self.alpha * (real_out ** 2 + aug_out ** 2).mean()
        cdiv_loss = aug_out.mean() - real_out.mean() 
        loss = reg_loss + cdiv_loss
        return loss

            
    def train_step(self, batch, update=True):
        """Train one batch

        Args:
            batch (dict): batch data
            update (bool, optional): whether to update networks. 
                Defaults to True.

        Returns:
            loss
        """
        # move inputs to device
        x = batch.clone().detach().requires_grad_(True).to(device)
        # compute losses
        if self.loss_type=='ssm_vr':
            v = torch.randn((self.n_slices,)+x.shape, dtype=x.dtype, device=device)
            v = v.view(-1, *v.shape[2:]) # (n_slices*b, 2)
            loss = self.ssm_vr_loss(x, v)
        elif self.loss_type=='ebm':
            loss = self.ebm_loss(x)
        
        # update model
        if update:
            # compute gradients
            loss.backward()
            # perform gradient updates
            grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.clipnorm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def train(self, dataset, batch_size):
        """Train one epoch

        Args:
            dataset (tf.data.Dataset): Tensorflow dataset
            batch_size (int): batch size

        Returns:
            np.ndarray: mean loss
        """        
        all_losses = []
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            loss = self.train_step(batch_data)
            self.num_gradsteps += 1
            all_losses.append(loss)
        m_loss = np.mean(all_losses).astype(np.float32)
        return m_loss

    def learn(
        self,
        train_dataset,
        n_epochs = 5,
        batch_size = 100
    ):
        """Train the model

        Args:
            train_dataset (Dataset): training dataset
            n_epochs (int, optional): number of epochs to train. Defaults to 5.
            batch_size (int, optional): batch size. Defaults to 100.

        Returns:
            self
        """

        for epoch in range(n_epochs):
            self.num_epochs += 1
            # train one epoch
            loss = self.train(train_dataset, batch_size)
            print(f"epoch:{epoch+1}, loss = ",loss)
            self.model.save('ebm')        

        return self