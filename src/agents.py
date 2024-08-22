import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from PIL import Image


from .utils import *
            
class SRGANAgent(object):
    def __init__(self, generator, discriminator, path, device, optimizers=None, criterions=None):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.optimizers = optimizers
        self.criterions = criterions
        self.path = path
        
        self.generator.to(self.device)
        if not (self.discriminator is None):
            self.discriminator.to(self.device)

        self.clock = TrainClock()
        self.train_tb, self.test_tb, self.valid_tb = exp_env(self.path)
        
        self.ckpt_path = os.path.join(self.path, "models")
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
            
        self.best_epoch = 1
        self.best_metric = 0

    def train(self, pretrain_epochs, train_epochs, train_loader, valid_loader, batch_size, save_frequency):
        
        loss_coefs = self.criterions['coeffs']
        
        ## PreTrain
        
        for epoch in range(pretrain_epochs):
            pbar = tqdm(train_loader)
            self.generator.train()
            local_history = {'g_loss': [], 'd_loss': [], 'PSNR': [], 'SSIM': []}
            
            for batch in pbar:
                gt = batch['GT'].to(self.device)
                lr = batch['LR'].to(self.device)

                output, _ = self.generator(lr)
                loss = self.criterions['MSE'](gt, output)

                self.optimizers['G'].zero_grad()
                loss.backward()
                self.optimizers['G'].step()
                
                psnr = self.psnr((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()
                ssim = 0
                
                local_history['g_loss'].append(loss.item())
                local_history['PSNR'].append(psnr)
                local_history['SSIM'].append(ssim)
                
                pbar.set_description("PRETRAIN EPOCH[{}/{}]".format(epoch+1, pretrain_epochs))
                pbar.set_postfix(g_loss=np.mean(local_history['g_loss']),
                                 PSNR=np.mean(local_history['PSNR']),
                                 SSIM=np.mean(local_history['SSIM']))
                
            pbar = tqdm(valid_loader, colour='green')    
            self.generator.eval()
            local_history = {'g_loss': [], 'd_loss': [], 'PSNR': [], 'SSIM': []}
            
            with torch.no_grad():
                for batch in pbar:
                    gt = batch['GT'].to(self.device)
                    lr = batch['LR'].to(self.device)

                    output, _ = self.generator(lr)
                    loss = self.criterions['MSE'](gt, output).item()

                    psnr = self.psnr((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()
                    ssim = self.ssim((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()

                    local_history['g_loss'].append(loss)
                    local_history['PSNR'].append(psnr)
                    local_history['SSIM'].append(ssim)

                    pbar.set_description("EVALUATION")
                    pbar.set_postfix(g_loss=np.mean(local_history['g_loss']),
                                     PSNR=np.mean(local_history['PSNR']),
                                     SSIM=np.mean(local_history['SSIM']))
                    
        if pretrain_epochs > 0:
            self.save_ckpt(0)
                
        ## Main Train
                
        for epoch in range(self.clock.epoch, train_epochs):
            self.generator.train()
            self.discriminator.train()
            
            pbar = tqdm(train_loader)
            
            local_history = {'g_loss': [], 'd_loss': [], 'PSNR': [], 'SSIM': []}
            
            for batch in pbar:
                gt = batch['GT'].to(self.device)
                lr = batch['LR'].to(self.device)

                ## Training Discriminator
                output, _ = self.generator(lr)
                fake_prob = self.discriminator(output)
                real_prob = self.discriminator(gt)
                
                real_label = torch.ones((real_prob.shape[0], 1)).to(self.device)
                fake_label = torch.zeros((fake_prob.shape[0], 1)).to(self.device)

                d_loss_real = self.criterions['BCE'](real_prob, real_label)
                d_loss_fake = self.criterions['BCE'](fake_prob, fake_label)
                d_loss = d_loss_real + d_loss_fake

                self.optimizers['G'].zero_grad()
                self.optimizers['D'].zero_grad()
                d_loss.backward()
                self.optimizers['D'].step()

                ## Training Generator
                output, _ = self.generator(lr)
                fake_prob = self.discriminator(output)
                real_label = torch.ones((fake_prob.shape[0], 1)).to(self.device)

                _percep_loss, hr_feat, sr_feat = self.criterions['VGG']((gt + 1.0) / 2.0, (output + 1.0) / 2.0)
                        
                percep_loss = loss_coefs['VGG'] * _percep_loss
                L2_loss = loss_coefs['MSE'] * self.criterions['MSE'](output, gt)
                adversarial_loss = loss_coefs['BCE'] * self.criterions['BCE'](fake_prob, real_label)
                total_variance_loss = loss_coefs['TV'] * self.criterions['TV'](loss_coefs['VGG'] * (hr_feat - sr_feat)**2)

                g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

                self.optimizers['G'].zero_grad()
                self.optimizers['D'].zero_grad()
                g_loss.backward()
                self.optimizers['G'].step()
                
                psnr = self.psnr((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()
                ssim = 0
                
                local_history['g_loss'].append(g_loss.item())
                local_history['d_loss'].append(d_loss.item())
                local_history['PSNR'].append(psnr)
                local_history['SSIM'].append(ssim)

                pbar.set_description("TRAIN EPOCH[{}/{}]".format(epoch+1, train_epochs))
                pbar.set_postfix(g_loss=np.mean(local_history['g_loss']),
                                 d_loss=np.mean(local_history['d_loss']),
                                 PSNR=np.mean(local_history['PSNR']),
                                 SSIM=np.mean(local_history['SSIM']))

                self.clock.tick()

            self.clock.tock()
            
            self.train_tb.add_scalar('g_loss', np.mean(local_history['g_loss']), global_step=epoch+1)
            self.train_tb.add_scalar('d_loss', np.mean(local_history['d_loss']), global_step=epoch+1)
            self.train_tb.add_scalar('PSNR', np.mean(local_history['PSNR']), global_step=epoch+1)
            self.train_tb.add_scalar('SSIM', np.mean(local_history['SSIM']), global_step=epoch+1)
            
            self.train_tb.add_image('HR', (gt[0] + 1.0) / 2.0, global_step=epoch+1)
            self.train_tb.add_image('LR', (lr[0] + 1.0) / 2.0, global_step=epoch+1)
            self.train_tb.add_image('SR', (output[0] + 1.0) / 2.0, global_step=epoch+1)
            
            self.generator.eval()
            self.discriminator.eval()
            local_history = {'g_loss': [], 'PSNR': [], 'SSIM': []}
            pbar = tqdm(valid_loader, colour='green')
            
            with torch.no_grad():
                for batch in pbar:
                    gt = batch['GT'].to(self.device)
                    lr = batch['LR'].to(self.device)

                    output, _ = self.generator(lr)
                    fake_prob = self.discriminator(output)
                    real_label = torch.ones((fake_prob.shape[0], 1)).to(self.device)

                    _percep_loss, hr_feat, sr_feat = self.criterions['VGG']((gt + 1.0) / 2.0, (output + 1.0) / 2.0)
                        
                    percep_loss = loss_coefs['VGG'] * _percep_loss
                    L2_loss = loss_coefs['MSE'] * self.criterions['MSE'](output, gt)
                    adversarial_loss = loss_coefs['BCE'] * self.criterions['BCE'](fake_prob, real_label)
                    total_variance_loss = loss_coefs['TV'] * self.criterions['TV'](loss_coefs['VGG'] * (hr_feat - sr_feat)**2)

                    g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

                    psnr = self.psnr((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()
                    if epoch % 5 == 0:
                        ssim = self.ssim((output + 1.0) / 2.0, (gt + 1.0) / 2.0).item()
                    else:
                        ssim = 0

                    local_history['g_loss'].append(g_loss.item())
                    local_history['PSNR'].append(psnr)
                    local_history['SSIM'].append(ssim)
                    
                    pbar.set_description("EVALUATION")
                    pbar.set_postfix(g_loss=np.mean(local_history['g_loss']),
                                     PSNR=np.mean(local_history['PSNR']),
                                     SSIM=np.mean(local_history['SSIM']))

            self.valid_tb.add_scalar('g_loss', np.mean(local_history['g_loss']), global_step=epoch+1)
            self.valid_tb.add_scalar('PSNR', np.mean(local_history['PSNR']), global_step=epoch+1)
            self.valid_tb.add_scalar('SSIM', np.mean(local_history['SSIM']), global_step=epoch+1)
            
            self.valid_tb.add_image('HR', (gt[0] + 1.0) / 2.0, global_step=epoch+1)
            self.valid_tb.add_image('LR', (lr[0] + 1.0) / 2.0, global_step=epoch+1)
            self.valid_tb.add_image('SR', (output[0] + 1.0) / 2.0, global_step=epoch+1)
            
            if np.mean(local_history['PSNR']) > self.best_metric:
                self.best_metric = np.mean(local_history['PSNR'])
                self.best_epoch = epoch+1
                    
            if self.clock.epoch % save_frequency == 0:
                self.save_ckpt(self.clock.epoch)
            
                
    def evaluate(self, valid_loader):
        fig, axes = plt.subplots(10,3,figsize=(30,100))

        pbar = tqdm(valid_loader, colour='green')
        local_history = {'PSNR': [], 'SSIM': []}

        with torch.no_grad():
            for i,batch in enumerate(pbar):
                gt = batch['GT'].to(self.device)
                lr = batch['LR'].to(self.device)

                output, _ = self.generator(lr)


                output = torch.clip(output, -1.0, 1.0)
                output = (output + 1.0) / 2.0
                gt = (gt + 1.0) / 2.0
                lr = (lr + 1.0) / 2.0

                psnr = self.psnr(output, gt).item()
                ssim = self.ssim(output, gt).item()

                output = np.uint8(output[0].detach().cpu().numpy().transpose((1,2,0)) * 255)
                lr = np.uint8(lr[0].detach().cpu().numpy().transpose((1,2,0)) * 255)
                gt = np.uint8(gt[0].detach().cpu().numpy().transpose((1,2,0)) * 255)

                local_history['PSNR'].append(psnr)
                local_history['SSIM'].append(ssim)

                Image.fromarray(output).save(os.path.join('datasets/results', f'sr_{i}.png'))
                Image.fromarray(gt).save(os.path.join('datasets/results', f'hr_{i}.png'))
                Image.fromarray(lr).save(os.path.join('datasets/results', f'lr_{i}.png'))

                if i < 10:
                    axes[i,0].imshow(gt)
                    axes[i,0].set_title('HR')
                    axes[i,1].imshow(lr)
                    axes[i,2].set_title('LR')
                    axes[i,2].imshow(output)
                    axes[i,2].set_title(f'SR PSNR={psnr}, SSIM={ssim}')

                pbar.set_description("EVALUATION")
                pbar.set_postfix(PSNR=np.mean(local_history['PSNR']),
                                 SSIM=np.mean(local_history['SSIM']))

    def predict(self, X):
        self.generator.eval()
        X = torch.tensor(X, dtype=torch.float32).reshape((1,)+X.shape)
        X = X.to(self.device)
        X = (X / 127.5) - 1
        X = self.generator(X)[0].detach().cpu()
        X = torch.clip(X, -1.0, 1.0)
        X = (X + 1.0) / 2.0
        return X*255

    def save_ckpt(self, n):
        save_path = os.path.join(self.ckpt_path, f"ckpt_epoch{n}.pth")
        print(f"Checkpoint saved at {save_path}")

        torch.save({
            'clock': self.clock.save(),
            'g_params': self.generator.cpu().state_dict(),
            'd_params': self.discriminator.cpu().state_dict(),
            'optimizer_g': self.optimizers['G'].state_dict(),
            'optimizer_d': self.optimizers['D'].state_dict()
        }, save_path)
            
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def load_ckpt(self, n):
        if type(n) == type(0):
            load_path = os.path.join(self.ckpt_path, f"ckpt_epoch{n}.pth")
        else:
            load_path = n
        if not os.path.exists(load_path):
            raise ValueError(f"Checkpoint {load_path} not exists.")

        checkpoint = torch.load(load_path)
        print(f"Checkpoint loaded from {load_path}")

        self.generator.load_state_dict(checkpoint['g_params'])
        self.discriminator.load_state_dict(checkpoint['d_params'])

        self.optimizers['G'].load_state_dict(checkpoint['optimizer_g'])
        self.optimizers['D'].load_state_dict(checkpoint['optimizer_d'])
        self.clock.load(checkpoint['clock'])

    def load_weights(self, n):
        if type(n) == type(0):
            load_path = os.path.join(self.ckpt_path, f"ckpt_epoch{n}.pth")
        else:
            load_path = n
        if not os.path.exists(load_path):
            raise ValueError(f"Weights {load_path} not exists.")

        checkpoint = torch.load(load_path)
        print(f"Weights loaded from {load_path}")

        self.generator.load_state_dict(checkpoint['g_params'])
