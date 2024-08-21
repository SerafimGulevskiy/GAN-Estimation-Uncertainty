import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# from torchvision.models import inception_v3
from scipy.linalg import sqrtm, norm
import numpy as np

def split_mnist_cats(dataset):
    # Group the test dataset by category (digits 0 to 9)
    category_data = {i: [] for i in range(10)}
    for img, label in dataset:
            category_data[label].append(img)
    return category_data

def split_mnist_loader_cats(dataloader, max_images = None):
    # Group the test dataset by category (digits 0 to 9)
    category_data = {i: [] for i in range(10)}
    for imgs, labels in dataloader:
        for img, label in zip(imgs, labels):
            if max_images:
                if len(category_data[label.item()]) < max_images:
                    category_data[label.item()].append(img)
            else:
                category_data[label.item()].append(img)
    return category_data

    
def calculate_fid(model, real_imgs, gen_imgs, device):
    # Load pre-trained InceptionV3 model
    # model().to(device=device)
    # model.eval();
    
    # Compute features for real images
    real_features = []
    for batch in real_imgs:
        batch = batch.to(device)
        features = model(batch).detach().cpu().numpy()
        for el in features:
            real_features.append(el)

    # Compute features for generated images
    gen_features = []
    for batch in gen_imgs:
        batch = batch.to(device)
        features = model(batch).detach().cpu().numpy()
        for el in features:
            gen_features.append(el)
            

    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    assert (
        mu_real.shape == mu_gen.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma_real.shape == sigma_gen.shape
    ), "Training and test covariances have different dimensions"

    # Compute FID score
    diff = mu_real - mu_gen
    cov_sqrt = sqrtm(sigma_real.dot(sigma_gen), disp=True)
    if np.iscomplexobj(cov_sqrt):
        #Return the real part of the complex argument
        cov_sqrt = cov_sqrt.real
        
    trace_cov = np.trace(sigma_real + sigma_gen - 2*cov_sqrt)
    fid_score = np.sum(diff**2) + trace_cov
    # print(np.sum(diff**2), np.trace(sigma_real + sigma_gen - 2*cov_sqrt))
    # trace_cov = covariance_distance(sigma_real, sigma_gen)
    # return fid_score, trace_cov
    return np.trace(sigma_real), mu_real

# Function to calculate the given distance metric
def covariance_distance(sigma1, sigma2):
    # Compute the trace of the dot product
    trace_r1_r2 = np.trace(np.dot(sigma1, sigma2))
    
    # Compute the Frobenius norm of each covariance matrix
    norm_r1 = norm(sigma1, 'fro')
    norm_r2 = norm(sigma2, 'fro')
    
    # Compute the distance
    distance = 1 - (trace_r1_r2 / (norm_r1 * norm_r2))
    
    return distance

def calculate_multiple_fid(generator, features_model, category_data, device):

    # Function to generate fake samples for a given class
    def generate_fake_samples(generator, num_samples, class_label, latent_dim = 100):
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.full((num_samples,), class_label).to(device)
        with torch.no_grad():
            fake_images = generator(z, labels).to(device)  # Move images to CPU for compatibility with torchvision datasets
        fake_images = fake_images.view(-1, 1, 28, 28)# Reshape each image to have size (1, 28, 28)
        return fake_images, labels
        
    fid_scores = {} 
    vfid_scores = {} 
    
    # Calculate FID score for each category
    for category, images in category_data.items():
        real_dataloader = DataLoader(images, batch_size=32, shuffle=False)
    
        
        # Generate fake images for this category
        fake_images = []
        # print(category)
        for i, real_batch in enumerate(real_dataloader):
            # print(len(real_batch))
            fake = generate_fake_samples(generator, len(real_batch), category)[0]
            # return fake
            fake_images.append(fake) 
        # return fake_images
        fake_images = torch.cat(fake_images, dim=0)
        # return fake_images
        fake_dataloader = DataLoader(fake_images, batch_size=32, shuffle=False)

        # Calculate FID score for this category
        fid_score, variance_fid = calculate_fid(features_model, real_dataloader, fake_dataloader, device)
        fid_scores[category] = fid_score
        vfid_scores[category] = variance_fid
        # Print FID score for this category
        # print(f'Category: {category}, FID: {fid_score}')
    
    return fid_scores, vfid_scores
        

# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, TensorDataset
# from scipy.linalg import sqrtm, norm
# import numpy as np
# def calculate_fid(model, real_loader, gen_loader, device):
#     """
#     Compute the FID score between two sets of images.
    
#     :param model: Pre-trained model used to extract features (e.g., InceptionV3).
#     :param real_loader: DataLoader for the real images.
#     :param gen_loader: DataLoader for the generated images.
#     :param device: Device on which the calculations are performed (e.g., 'cuda' or 'cpu').
#     :return: The FID score and the variance for diagnostic purposes.
#     """
#     # Set the model to evaluation mode
#     # model.to(device)
#     # model.eval()
    
#     # Compute features for real images
#     real_features = []
#     for imgs, _ in real_loader:  # Make sure you iterate over the DataLoader
#         imgs = imgs.to(device)  # Ensure imgs is a tensor
#         features = model(imgs, _).detach().cpu().numpy()
#         real_features.extend(features)  # Append all features to the list
    
#     # Compute features for generated images
#     gen_features = []
#     for imgs, _ in gen_loader:  # Ensure you iterate over the DataLoader
#         imgs = imgs.to(device)  # Convert to the correct device
#         features = model(imgs, _).detach().cpu().numpy()
#         gen_features.extend(features)  # Append all features to the list
    
#     # Calculate the mean and covariance for both sets of features
#     mu_real = np.mean(real_features, axis=0)
#     sigma_real = np.cov(real_features, rowvar=False)
#     mu_gen = np.mean(gen_features, axis=0)
#     sigma_gen = np.cov(gen_features, rowvar=False)
    
#     # Calculate the FID score
#     diff = mu_real - mu_gen
#     cov_sqrt = sqrtm(sigma_real.dot(sigma_gen), disp=True)
#     if np.iscomplexobj(cov_sqrt):
#         cov_sqrt = cov_sqrt.real  # Take only the real part if there are imaginary components
    
#     trace_cov = np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
#     fid_score = np.sum(diff**2) + trace_cov
    
#     return fid_score, trace_cov

# def split_mnist_cats(dataset):
#     # Group the dataset by category (digits 0 to 9) and retain labels
#     category_data = {i: [] for i in range(10)}
#     for img, label in dataset:
#         category_data[label].append((img, label))  # Store image with its label
#     return category_data

# def split_mnist_loader_cats(dataloader, max_images=None):
#     # Group the dataloader by category and retain labels
#     category_data = {i: [] for i in range(10)}
#     for imgs, labels in dataloader:
#         for img, label in zip(imgs, labels):
#             if max_images is None or len(category_data[label.item()]) < max_images:
#                 category_data[label.item()].append((img, label))  # Store image with its label
#     return category_data

# # Function to generate fake samples for a given class
# def generate_fake_samples(generator, num_samples, class_label, latent_dim=100, device="cpu"):
#     z = torch.randn(num_samples, latent_dim).to(device)
#     labels = torch.full((num_samples,), class_label).to(device)
#     with torch.no_grad():
#         fake_images = generator(z, labels)  # Generate fake images
#     fake_images = fake_images.view(-1, 1, 28, 28)  # Reshape each image
#     return fake_images, labels  # Return images with their labels

# # Now, let's update the `calculate_multiple_fid` function to include the labels for the fake dataset
# def calculate_multiple_fid(generator, features_model, category_data, device):
#     fid_scores = {}
#     vfid_scores = {}

#     # Calculate FID score for each category
#     for category, images in category_data.items():
#         # Create a dataloader for the real images
#         real_images = [x[0] for x in images]  # Extract the images
#         real_labels = [x[1] for x in images]  # Extract the labels
#         real_dataloader = DataLoader(TensorDataset(torch.stack(real_images), torch.tensor(real_labels)), batch_size=32, shuffle=False)

#         # Generate fake images for this category with labels
#         fake_images = []
#         for real_batch, _ in real_dataloader:
#             fake_batch, _ = generate_fake_samples(generator, len(real_batch), category, device=device)
#             fake_images.append(fake_batch)

#         fake_images = torch.cat(fake_images, dim=0)
#         fake_labels = torch.full((len(fake_images),), category, dtype=torch.long)  # All generated images share the same label
#         fake_dataloader = DataLoader(TensorDataset(fake_images, fake_labels), batch_size=32, shuffle=False)

#         # Calculate FID score for this category
#         fid_score, variance_fid = calculate_fid(features_model, real_dataloader, fake_dataloader, device)
#         fid_scores[category] = fid_score
#         vfid_scores[category] = variance_fid

#     return fid_scores, vfid_scores
