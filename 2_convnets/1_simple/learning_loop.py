import os
import torch
from torch.autograd import Variable
from tqdm import tqdm

class LearningLoop(object):
    def __init__(self, model, optimizer, loss_funct, 
               train_loader=None, valid_loader=None, test_loader=None, 
               checkpoint_path=None, checkpoint_load=None, cuda_flag=False):
   
    # are checkpoints required?
        if checkpoint_path:
            if not os.path.isdir(checkpoint_path) :
                os.mkdir(checkpoint_path)        
            self.checkpoint_path = checkpoint_path
            self.checkpoint_filename = os.path.join(self.checkpoint_path, 'checkpoint_{:03d}.pt')
            self.checkpoint_flag = True
        else:
            self.checkpoint_flag = False
      
        self.cuda_flag = cuda_flag
        self.model = model
        self.optimizer = optimizer
        self.loss_funct = loss_funct
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.history = {'train_loss': [], 'valid_loss': []}
        self.ix_epoch = 0
        self.epochs_no_improv = 0
        self.best_val_loss = float('inf')
        
        self.tqdm_ncols = 80
        
        # if a specific checkpoint (int) is indicated and it exists, restart from that checkpoint
        if checkpoint_load and self.checkpoint_flag:
            ckpt_filename = self.checkpoint_filename.format(checkpoint_load)
            self.load_state_file(ckpt_filename)
     
    def train(self, n_epochs=1, patience=5):
        while self.ix_epoch < n_epochs and self.epochs_no_improv < patience:
            self.ix_epoch += 1
            ## TRAINING
            self.model.train()
            train_loss = 0.0
            train_acc  = 0.0      
            train_bar_text = 'Training data  '
            
            n_batch = self.train_loader.__len__()
            for batch in tqdm(self.train_loader, desc=train_bar_text, ncols=self.tqdm_ncols, unit='batch'):
                new_train_loss, new_train_acc = self.train_batch(batch)
                train_loss += new_train_loss
                train_acc  += new_train_acc
            
            train_loss = train_loss / n_batch
            train_acc  = train_acc / n_batch
            self.history['train_loss'].append(train_loss)
      
            ## VALIDATION
            self.model.eval()
            valid_loss = 0.0
            valid_acc  = 0.0     
            valid_bar_text = 'Validation data'
          
            n_batch = self.valid_loader.__len__()
            for batch in tqdm(self.valid_loader, desc=valid_bar_text, ncols=self.tqdm_ncols, unit='batch'):
                new_valid_loss, new_valid_acc = self.eval_batch(batch)
                valid_loss += new_valid_loss
                valid_acc  += new_valid_acc

            valid_loss = valid_loss / n_batch
            valid_acc  = valid_acc / n_batch
            self.history['valid_loss'].append(valid_loss)

            print('Train: loss: {:.4f}    acc: {:.4f} '.format(train_loss, train_acc))
            print('Valid: loss: {:.4f}    acc: {:.4f} '.format(valid_loss, valid_acc))

            # has the model improved?
            if valid_loss < self.best_val_loss:
                if self.checkpoint_flag:
                    ckpt_filename = self.checkpoint_filename.format(self.ix_epoch)
                    self.save_state_file(ckpt_filename)
                self.epochs_no_improv = 0
                self.best_val_loss = valid_loss
            else:
                self.epochs_no_improv += 1

        # save final model
        self.save_state_file('./final_state.pt')

    def test(self):
        # TESTING
        self.model.eval()
        test_loss = 0.0
        test_acc  = 0.0     
        test_bar_text ='Testing   data'
    
        n_batch = self.test_loader.__len__()
        for batch in tqdm(self.test_loader, desc=test_bar_text, ncols=self.tqdm_ncols, unit='batch'):
            new_test_loss, new_test_acc = self.eval_batch(batch)
            test_loss += new_test_loss
            test_acc  += new_test_acc

        test_loss = test_loss / n_batch
        test_acc  = test_acc / n_batch
    
        print('Test : loss: {:.4f}    acc: {:.4f} '.format(test_loss, test_acc))

    def train_batch(self, batch):
        x, y = batch
        y = y.squeeze()

        # move to GPU if required
        if self.cuda_flag:
            x = x.cuda()
            y = y.cuda()
      
        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)
        
        y_hat = self.model.forward(x)         # foward pass
        loss = self.loss_funct(y_hat, y)      # compute loss
        self.optimizer.zero_grad()            # set gradients to zero
        loss.backward()                       # backpropagation
        self.optimizer.step()                 # step to update parameters
        acc = self.accuracy(y_hat, y)         # accuracy
        
        return loss.item(), acc

    def eval_batch(self, batch):
        x, y = batch
        y = y.squeeze()
    
        # move to GPU if required
        if self.cuda_flag:
            x = x.cuda()
            y = y.cuda()

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)
    
        y_hat = self.model.forward(x)         # foward pass
        loss = self.loss_funct(y_hat, y)      # compute loss
        acc = self.accuracy(y_hat, y)         # accuracy
        
        return loss.item(), acc

    def save_state_file(self, filename):
        # Save state file
        ckpt = {'model_state':      self.model.state_dict(),
                'optimizer_state':  self.optimizer.state_dict(),
                'loss_funct':       self.loss_funct,
                'history':          self.history,
                'ix_epoch':         self.ix_epoch,
                'epochs_no_improv': self.epochs_no_improv,
                'best_val_loss':    self.best_val_loss}
        torch.save(ckpt, filename)
        print('State file: {} saved.'.format(os.path.basename(filename)))
    
    def load_state_file(self, filename):
        print('')
        if os.path.isfile(filename):
            # Load state file
            ckpt = torch.load(filename)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.loss_funct = ckpt['loss_funct']
            self.history = ckpt['history']
            self.ix_epoch = ckpt['ix_epoch']
            self.epochs_no_improv = ckpt['epochs_no_improv']
            self.best_val_loss = ckpt['best_val_loss']
            print('State file: {} loaded.'.format(os.path.basename(filename)))
        else:
            print('Not found checkpoint file {}'.format(os.path.basename(filename)))

    def accuracy_2(self, y_hat, y):
        y_hat_v = torch.max(y_hat, dim=1)[1]
        y_v = torch.max(y, dim=1)[1]
        acc = torch.mean((y_hat_v == y_v).type(torch.FloatTensor))
        return acc.item()

    def accuracy(self, y_hat, y):
        y_hat_v = torch.max(y_hat, dim=1)[1]
        acc = torch.mean((y_hat_v == y).type(torch.FloatTensor))
        return acc.item()

    def print_params_norms(self):
        norm = 0.0
        for params in list(self.model.parameters()):
            norm+=params.norm(2).data[0]
        print('Sum of weights norms: {}'.format(norm))
    
    def print_params_count(self):
        count = 0
        for params in list(self.model.parameters()):
            count+=params.numel()
        print('Number of parameters: {}'.format(count))

    def print_grad_norms(self):
        norm = 0.0
        for i, params in enumerate(list(self.model.parameters())):
            norm+=params.grad.norm(2).data[0]
        print('Sum of grads norms: {}'.format(norm))


