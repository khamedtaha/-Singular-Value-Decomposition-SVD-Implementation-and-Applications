import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# TODO: When ready, import your SVD decomposition function.
# Uncomment the line below when ready.
from svd_lab import svd_decomposition, low_rank_svd

def image_compression(U, Sigma, V, k):
    """
    Given the precomputed SVD components and a specified k,
    returns the reconstructed image using the top-k singular values.
    
    Parameters:
    - U: Left singular matrix.
    - Sigma: Diagonal matrix of singular values.
    - V: Right singular matrix.
    - k: Number of singular values/features to keep.
    
    Returns:
    - A_reconstructed: The reconstructed image.
    """
    # Extract the top-k components.
    U_k, Sigma_k, V_k = low_rank_svd(U ,Sigma, V ,  k)
    
    # Reconstruct the image using the truncated SVD components.
    A_reconstructed =   U_k @ Sigma_k @ V_k

    return A_reconstructed

def main():
    images_path = r'application-datasets\app1-image-compression-and-reconstruction'

    # List of test images.
    test_images = ['webb-cartwheel-galaxy.png', 'tarantula-nebula.jpg']

    # TODO: Choose an example image to test.
    test_image = test_images[0]

    image_path = os.path.join(images_path, test_image)

    # Opening the image, converting it to grayscale, and transforming it into a numpy array.
    img = Image.open(image_path).convert('L')  # 'L' mode is for grayscale.
    A = np.array(img, dtype=float)

    m, n = A.shape
    print(f'm = {m}')
    print(f'n = {n}')
    
    # Compute the SVD of the image using your svd_decomposition function.
    U, Sigma, V = svd_decomposition(A)

    print('U:', U.shape)
    print('Sigma:', Sigma.shape)
    print('V:', V.shape)

    # k is the number of singular values (latent features) to retain.
    # Define the different values of k.
    k_values = [25, 50, 100]

    # Plot the original image and its reconstructions.
    num_plots = len(k_values) + 1  # One for the original image plus one per k value.
    plt.figure(figsize=(15, 5))
    
    # Display the original image.
    plt.subplot(1, num_plots, 1)
    plt.imshow(A, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Loop over each k, reconstruct the image, and display.
    for i, k in enumerate(k_values, start=2):
        # Reconstruct the image using the current k value.
        A_reconstructed = image_compression(U, Sigma, V, k)
        plt.subplot(1, num_plots, i)
        plt.imshow(A_reconstructed, cmap='gray')
        plt.title(f'Reconstructed Image (k={k})')
        plt.axis('off')
    
    plt.tight_layout()
    # Saving the results
    output_file = f'{os.path.splitext(test_image)[0]}-results.png'
    output_path = os.path.join(images_path, output_file)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    # Displaying the results
    plt.show()

if __name__ == '__main__':
    main()
