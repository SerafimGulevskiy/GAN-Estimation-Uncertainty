import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        
        def block(in_filters, out_filters,
                  normalize=False,
                  kernel_size = 3, stride = 2, padding = 1,
                  dropout = 0.3):
            
                layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
                
                if normalize:
                    layers.append(nn.BatchNorm2d(out_filters))
                    
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            
        self.model = nn.Sequential(
            *block(1, 32, normalize = True),
            *block(32, 64, normalize = True),
            *block(64, 128, normalize = True),
            *block(128, 512, normalize = True),
            nn.Conv2d(512, 10, 2, padding=0),
            nn.Flatten()
        )
        
    def forward(self, img):
        out = self.model(img)
        return out
    
class CNN2(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        
        def block(in_filters, out_filters,
                  normalize=False,
                  kernel_size = 3, stride = 2, padding = 1,
                  dropout = 0.3):
            
                layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
                
                if normalize:
                    layers.append(nn.BatchNorm2d(out_filters))
                    
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                layers.append(nn.Dropout(dropout))
                
                # layers.append(nn.LeakyReLU(0.2, inplace=True))
                # layers.append(nn.GELU())
                return layers
            
        self.model = nn.Sequential(
            *block(1, 32, normalize = True),
            *block(32, 64, normalize = True),
            *block(64, 128, normalize = True),
            *block(128, 512, normalize = True),
            nn.Conv2d(512, 10, 2, padding=0),
            nn.Flatten()
        )
        
    def forward(self, img):
        out = self.model(img)
        return out
    
   
class CNN3(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        
        def block(in_filters, out_filters,
                  normalize=False,
                  kernel_size = 3, stride = 2, padding = 1,
                  dropout = 0.1):
            
                layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
                
                if normalize:
                    layers.append(nn.BatchNorm2d(out_filters))
                    
                layers.append(nn.LeakyReLU(0.2, inplace=True))

                layers.append(nn.Dropout(dropout))
                
                # layers.append(nn.LeakyReLU(0.2, inplace=True))
                # layers.append(nn.GELU())
                return layers
            
        self.model = nn.Sequential(
            *block(1, 32, normalize = True),
            *block(32, 64, normalize = True),
            *block(64, 128, normalize = True),
            *block(128, 256, normalize = True),
            *block(256, 512, normalize = True),
            nn.Conv2d(512, 10, 1, padding=0),#mlp
            nn.Flatten()
        )
        
    def forward(self, img):
        out = self.model(img)
        return out
    
    


class CNNClassifierWrapper:
    def __init__(self, model, layer_index=-3, use_global_pooling=False):
        self.model = model
        self.activation = {}
        self.base_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=(0.5,), std=(0.5,))
                        ])
        self.layer_index = layer_index
        self.model.model[self.layer_index].register_forward_hook(self.get_activation('extract'))
        self.use_global_pooling = use_global_pooling
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def preprocess_image(self, image):
        # Apply transformations to the image
        preprocessed_image = self.base_transform(image)
        # .unsqueeze(0)  # Add batch dimension
        return preprocessed_image

    def __call__(self, image, transform = False):
        
        # Preprocess the image
        if transform:
            preprocessed_image = self.preprocess_image(image)
        output = self.model(image)
        output = self.activation['extract']
        
        if self.use_global_pooling:
            output = F.adaptive_avg_pool2d(output, (1, 1))

        output = output.view(output.size(0), -1)  # Flatten to (batch_size, num_channels)
        
        return output