import os
import torch
from torch.autograd import Variable
from tqdm import tqdm

class LearningLoop(object):
  def __init__(self, model_name, model, optimizer, loss_funct,
               train_loader=None, valid_loader=None, test_loader=None,
               ckpt_path=None, ckpt_load=None, cuda_flag=False):

    self.model_name = model_name
    self.model = model
    self.optimizer = optimizer
    self.loss_funct = loss_funct
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.test_loader = test_loader
    self.cuda_flag = cuda_flag

  # are checkpoints required?
    if ckpt_path:
      ckpt_path = ckpt_path + '_' + self.model_name
      if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
      self.ckpt_path = ckpt_path
      self.ckpt_filename = os.path.join(self.ckpt_path, 'ckpt_{:03d}.pt')
      self.ckpt_flag = True
    else:
      self.ckpt_flag = False

    self.history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    self.ix_epoch = 0
    self.epochs_no_improv = 0
    self.best_val_acc = 0
    self.tqdm_ncols = 80

    # if a specific checkpoint (ckpt_load) is indicated and it exists, restart from that checkpoint
    if ckpt_load and self.ckpt_flag:
      ckpt_filename = self.ckpt_filename.format(ckpt_load)
      self.load_state_file(ckpt_filename)
    return

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
      self.history['train_acc'].append(train_acc)

      ## VALIDATION
      self.model.eval()
      valid_loss = 0.0
      valid_acc  = 0.0
      valid_bar_text = 'Validation data'

      n_batch = self.valid_loader.__len__()
      for batch in tqdm(self.valid_loader, desc=valid_bar_text, ncols=self.tqdm_ncols, unit='batch'):
        new_valid_loss, new_valid_acc, _ = self.eval_batch(batch)
        valid_loss += new_valid_loss
        valid_acc  += new_valid_acc

      valid_loss = valid_loss / n_batch
      valid_acc  = valid_acc / n_batch
      self.history['valid_loss'].append(valid_loss)
      self.history['valid_acc'].append(valid_acc)

      print('')
      print('Train: loss: {:.4f}    acc: {:.4f} '.format(train_loss, train_acc))
      print('Valid: loss: {:.4f}    acc: {:.4f} '.format(valid_loss, valid_acc))

      # has the accuracy in the validation dataset improved?
      if valid_acc > self.best_val_acc:
        if self.ckpt_flag:
          ckpt_filename = self.ckpt_filename.format(self.ix_epoch)
          self.save_state_file(ckpt_filename)
        self.epochs_no_improv = 0
        self.best_val_acc = valid_acc
      else:
        self.epochs_no_improv += 1

    # load last ckpt file
    self.load_state_file(ckpt_filename)
    # save final state 
    self.save_state_file('./final_state_' + self.model_name + '.pt')
    return

  def test(self):
    # TESTING
    self.model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    y_hat_batchs = []
    test_bar_text ='Testing   data'

    n_batch = self.test_loader.__len__()
    for batch in tqdm(self.test_loader, desc=test_bar_text, ncols=self.tqdm_ncols, unit='batch'):
      new_test_loss, new_test_acc, new_y_hat = self.eval_batch(batch)
      test_loss += new_test_loss
      test_acc  += new_test_acc
      y_hat_batchs.append(new_y_hat)

    test_loss = test_loss / n_batch
    test_acc  = test_acc / n_batch
    y_hat = torch.cat(y_hat_batchs, 0)

    print('Test : loss: {:.4f}    acc: {:.4f} '.format(test_loss, test_acc))
    return y_hat

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

    y_hat_v = torch.max(y_hat, dim=1)[1]  # y_hat as 1D tensor

    return loss.item(), acc, y_hat_v

  def save_state_file(self, filename):
    # Save state file
    ckpt = {'model_name':       self.model_name,
            'model_state':      self.model.state_dict(),
            'optimizer_state':  self.optimizer.state_dict(),
            'loss_funct':       self.loss_funct,
            'history':          self.history,
            'ix_epoch':         self.ix_epoch,
            'epochs_no_improv': self.epochs_no_improv,
            'best_val_acc':     self.best_val_acc}
    torch.save(ckpt, filename)
    print('State file: {} saved.'.format(os.path.basename(filename)))
    return

  def load_state_file(self, filename):
    print('')
    if os.path.exists(filename):
      # Load state file
      ckpt = torch.load(filename)
      self.model_name = ckpt['model_name']
      self.model.load_state_dict(ckpt['model_state'])
      self.optimizer.load_state_dict(ckpt['optimizer_state'])
      self.loss_funct = ckpt['loss_funct']
      self.history = ckpt['history']
      self.ix_epoch = ckpt['ix_epoch']
      self.epochs_no_improv = ckpt['epochs_no_improv']
      self.best_val_acc = ckpt['best_val_acc']
      print('State file: {} loaded.'.format(os.path.basename(filename)))
    else:
      print('Not found checkpoint file {}'.format(os.path.basename(filename)))
    return

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
