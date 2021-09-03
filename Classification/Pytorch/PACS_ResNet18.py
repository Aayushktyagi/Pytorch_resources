import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import PIL
# from torchlars import LARS
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
print(torch.__version__)
from sklearn.metrics import accuracy_score
import torch.optim as optim
import tqdm
torch.manual_seed(1)
np.random.seed(1)
##################################################### Training f_theta network ###########################################

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CHECKPOINT_DIR = "./models/Vae_128_arnab_allsource/"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 350
FEATURE_DIM = 256
IMAGE_SIZE = 224
CLASSES = 7
LR = 1

src_path =['photo', 'art_painting', 'cartoon']
target_path = ['sketch']

# parameters
resnet_train = False
classifier_load = True
train_clf = False
train_vae = False

# data paths 
image_data_path = '/data/pacs_data'
image_label_path = 'data/pacs_label'
image_list_train_photo = '/pacs_label/photo_train_kfold.txt'
image_list_train_cartoon = '/data/pacs_label/cartoon_train_kfold.txt'
image_list_train_art ='/data/pacs_label/art_painting_train_kfold.txt'
image_list_valid_photo = '/DG/data/pacs_label/photo_crossval_kfold.txt'
image_list_valid_cartoon = '/DG/data/pacs_label/cartoon_crossval_kfold.txt'
image_list_valid_art = '/DG/data/pacs_label/art_painting_crossval_kfold.txt'
image_list_test_sketch = '/DG/data/pacs_label/sketch_test_kfold.txt'

# Weights 
resnet_weights = '/code/models/resnet_rot_30_invcanedge05/epoch_pacs_resnet_rot30_invcanedge05_349.pt'
classifier_weights = '/DG/code/models/resnet_rot_30_invcanedge05/resnetrot30classifierep50lr001.pt'
classifier_weight_finetuned='/DG/code/vae/single_source/AutoEncoder/models/Vae_128_arnab/classifier_finetuned_vae128.pt'
vae_weights = '/DG/code/vae/single_source/AutoEncoder/models/Vae_128_arnab_allsource/VAEepoch_pacs_resnet_1300.pt'

class PACSDataset(Dataset):
  """ PACS (Photo Art Cartoon Sketch) Dataset """

  def __init__(self, root_dir, image_size, domains=None, transform = None, img_lists = None):
    """
    Arguments:
      img_lists (list of strings): Path to the image list describing the split
      root_dir (string): Path to the root directory of the different domain folders 
      domains (list of strings): List of domains to read the data from. If None,
        all domains are read
      image_size (integer): Image size to resize the images to
    """
    self.root_dir = root_dir
    self.img_lists = img_lists
    if root_dir[-1] != "/":
      self.root_dir = self.root_dir + "/"
    
    self.categories = ['giraffe', 'horse', 'guitar', 'person', 'dog', 'house', 'elephant']

    if domains is None:
      self.domains = ["photo", "sketch", "art_painting", "cartoon"]
    else:
      self.domains = domains
    
    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform
    # make a list of all the files in the root_dir
    # and read the labels
    self.img_files = []
    self.labels = []
    self.domain_labels = []
    if img_lists is None:
      for domain in self.domains:
        for category in self.categories:
          for image in os.listdir(self.root_dir+domain+'/'+category):
            self.img_files.append(image)
            self.labels.append(self.categories.index(category))
            self.domain_labels.append(self.domains.index(domain))
    else:
      for img_list in img_lists:
        with open(img_list) as img_files:
          for line in img_files:
            img_path, label = line.split()
            domain, category, image = img_path.split("/")
            self.img_files.append(image)
            self.labels.append(self.categories.index(category))
            self.domain_labels.append(self.domains.index(domain))

  def __len__(self):
    return len(self.img_files)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    img_path = self.root_dir + self.domains[self.domain_labels[idx]] + "/" + self.categories[self.labels[idx]] + "/" + self.img_files[idx]
    
    image = PIL.Image.open(img_path)
    label = self.labels[idx]

    return self.transform(image), label


import cv2
import numpy as np

np.random.seed(0)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample
    
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
class cannyedge(object):
    def __init__(self, threshold1 = 100, threshold2 = 200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
    def __call__(self, sample):
        prob = np.random.random_sample()
        if prob < 0.1:
            sample = np.array(sample)
            edges = cv2.Canny(image=sample, threshold1=self.threshold1, threshold2=self.threshold2) # Canny Edge Detection
            edge3c = np.zeros_like(sample)
            edge3c[:,:,0] = edges
            edge3c[:,:,1] = edges
            edge3c[:,:,2] = edges
            invedge = cv2.bitwise_not(edge3c)

            return invedge
        else:
            return sample

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=IMAGE_SIZE),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.5),
                                              transforms.RandomGrayscale(p=0.5),
                                              transforms.RandomRotation(30),
#                                               GaussianBlur(kernel_size=int(21)), 
                                              cannyedge(),
                                              transforms.ToTensor(),
                                              
#                                               AddGaussianNoise(mean=0, std=0.2),
] )
training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo, image_list_train_cartoon, image_list_train_art])

validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo', 'art_painting', 'cartoon'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo,image_list_valid_cartoon, image_list_valid_art])

training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)


class FNet_PACS_ResNet(nn.Module):
 
  def __init__(self, hidden_layer_neurons, output_latent_dim):
    super(FNet_PACS_ResNet, self).__init__()
    resnet = resnet18(pretrained=True)
    
    self.resnet = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(resnet.fc.in_features,  hidden_layer_neurons)
    self.fc2 = nn.Linear(hidden_layer_neurons, output_latent_dim)
   
  def forward(self, x):
    x = self.resnet(x)
    x = x.squeeze()

    x = self.fc1(x)
    x = F.leaky_relu(x, negative_slope=0.2)

    x = self.fc2(x)
    return x



def train_step(x, labels, model, optimizer, tau):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    z = model(x)

    # Calculate loss
    z = F.normalize(z, dim=1)
    pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
    logits = torch.flatten(torch.matmul(z, z.t())) / tau
    loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
    pred = torch.sigmoid(logits)   # whether two images are similar or not
    accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])

    # Perform train step
    #optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def validation_step(x, labels, model, tau):
    model.eval()
    with torch.no_grad():
        z = model(x)

        # Calculate loss
        z = F.normalize(z, dim=1)
        pairwise_labels = torch.flatten(torch.matmul(labels, labels.t()))
        logits = torch.flatten(torch.matmul(z, z.t())) / tau
        loss = F.binary_cross_entropy_with_logits(logits, pairwise_labels)
        pred = torch.sigmoid(logits)   # whether two images are similar or not
        accuracy = (pred.round().float() == pairwise_labels).sum()/float(pred.shape[0])    
    return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

def training_loop(model, training_dataloader, validation_dataloader, optimizer, tau=0.1, epochs=200, device=None):
    epoch_wise_loss = []
    epoch_wise_acc = []
    for epoch in range(EPOCHS):
        step_wise_train_loss = []
        step_wise_train_acc = []
        step_wise_val_loss = []
        step_wise_val_acc = []

        for image_batch, labels in training_dataloader:
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, CLASSES).float()
            loss, accuracy = train_step(image_batch, labels_onehot, model, optimizer, tau)
            step_wise_train_loss.append(loss)
            step_wise_train_acc.append(accuracy)

        for image_batch, labels in validation_dataloader:
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(device), labels.to(device)
            labels_onehot = F.one_hot(labels, CLASSES).float()
            loss, accuracy = validation_step(image_batch, labels_onehot, model, tau)
            step_wise_val_loss.append(loss)
            step_wise_val_acc.append(accuracy)

        if (epoch+1)%10 == 0 and epoch>200:
            torch.save({'epoch' : epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss}, CHECKPOINT_DIR+"epoch_pacs_resnet_rot30_invcanedge10_"+str(epoch)+".pt")
        epoch_wise_loss.append(np.mean(step_wise_train_loss))
        epoch_wise_acc.append(np.mean(step_wise_train_acc))
        print("epoch: {} training loss: {:.3f} training accuracy: {:.3f} validation loss: {:.3f} validation accuracy: {:.3f}".format(epoch + 1, np.mean(step_wise_train_loss), np.mean(step_wise_train_acc), np.mean(step_wise_val_loss), np.mean(step_wise_val_acc)))

    return epoch_wise_loss, epoch_wise_acc, model

if resnet_train:
    model = FNet_PACS_ResNet(512, FEATURE_DIM)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(dev)
    optimizer = LARS(torch.optim.SGD(model.parameters(), lr=LR))
    epoch_wise_loss, epoch_wise_acc, model = training_loop(model, training_dataloader, validation_dataloader, optimizer, tau=0.1, epochs=EPOCHS, device=dev)
else:

    model = FNet_PACS_ResNet(512, FEATURE_DIM)
    checkpoint = torch.load(resnet_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(dev)

    model.eval()

if classifier_load:
    'Creating model and load weights'
    layers = []
    layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
    layers.append(nn.Linear(FEATURE_DIM, CLASSES))
    classifier = torch.nn.Sequential(*layers).to(dev)
    classifier = classifier.to(dev)
    classifier.load_state_dict(torch.load(classifier_weights))
else:
    'Creating model'
    layers = []
    layers.append(nn.Linear(FEATURE_DIM, FEATURE_DIM))
    layers.append(nn.Linear(FEATURE_DIM, CLASSES))
    classifier = torch.nn.Sequential(*layers).to(dev)
    classifier = classifier.to(dev)

CELoss = nn.CrossEntropyLoss()

if train_clf:
    opt = torch.optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(50):
        step_wise_loss = []
        step_wise_accuracy = []

        step_wise_val_loss = []
        step_wise_val_accuracy = []

        classifier.train()
        for image_batch, labels in (training_dataloader):
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(dev), labels.to(dev)

            # zero the parameter gradients
            opt.zero_grad()

            z = model(image_batch).to(dev)
            pred = classifier(z)
            loss = CELoss(pred, labels)
            accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
            loss.backward()
            opt.step()

            step_wise_loss.append(loss.detach().cpu().numpy())
            step_wise_accuracy.append(accuracy.detach().cpu().numpy())

        classifier.eval()
        for image_batch, labels in (validation_dataloader):
            image_batch = image_batch.float()
            if dev is not None:
                image_batch, labels = image_batch.to(dev), labels.to(dev)

            z = model(image_batch).to(dev)
            pred = classifier(z)
            loss = CELoss(pred, labels)
            accuracy = (pred.argmax(dim=1) == labels).float().sum()/pred.shape[0]
            loss.backward()
            opt.step()

            step_wise_val_loss.append(loss.detach().cpu().numpy())
            step_wise_val_accuracy.append(accuracy.detach().cpu().numpy())

        print("Epoch " + str(epoch) + " Loss " + str(np.mean(step_wise_loss)) + " Accuracy " + str(np.mean(step_wise_accuracy)))
        print("Val loss " + str(np.mean(step_wise_val_loss)) + " Val accuracy " + str(np.mean(step_wise_val_accuracy)))

#Load single source data again
photo_training_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[ image_list_train_photo])

photo_validation_dataset = PACSDataset(image_data_path, IMAGE_SIZE, 
                               ['photo'], 
                               transform=data_transforms, 
                               img_lists=[image_list_valid_photo])

photo_training_dataloader = DataLoader(photo_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)
photo_validation_dataloader = DataLoader(photo_validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (photo_training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))

classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (training_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))

test_dataset = PACSDataset( image_data_path, IMAGE_SIZE, 
                           ['sketch'], 
                           img_lists=[image_list_test_sketch])
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 1)


classifier.eval()
preds = []
true = []
with torch.no_grad():
    for image_batch, labels in (test_dataloader):
        image_batch = image_batch.float()
        if dev is not None:
            image_batch, labels = image_batch.to(dev), labels.to(dev)
        z = model(image_batch).to(dev)
        pred = classifier(z)
        preds.append(pred.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        accuracy = (pred.argmax(dim=1) == labels).float().sum()
        #       step_wise_accuracy.append(accuracy.detach().cpu().numpy())
preds = np.concatenate(preds, axis=0)
true = np.concatenate(true, axis=0)
print(accuracy_score(preds, true))
