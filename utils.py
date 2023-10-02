"""
Insert your utility functions in this module
"""
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def load_faces():  # you can give other arguments if you wish
    # do not modify `fetch_lfw_people` default arguments
    faces_dataset = fetch_lfw_people(resize=0.7, min_faces_per_person=50, color=False)

    # TODO you can return anything that you want
    return faces_dataset


def pca_faces(X, optimal_num_components):
    reduced_images = []
    pca = PCA(n_components=optimal_num_components)
    pca.fit(X)
    X_pca= pca.transform(X)

    X_back = pca.inverse_transform(X_pca)
    reduced_images.append(X_back)
    return reduced_images


def plot_pca_faces(X, image_shape, optimal_num_components):
    reduced_images = pca_faces(X, optimal_num_components)

    # plot the first three images in the test set:
    fix, axes = plt.subplots(4, 2, figsize=(7, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in enumerate(axes):
        # plot original image
        ax[0].imshow(X[i].reshape(image_shape),
                     vmin=0, vmax=1, cmap="gray")
        # plot the four back-transformed images
        for a, X_back in zip(ax[1:], reduced_images):
            a.imshow(X_back[i].reshape(image_shape), vmin=0, vmax=1, cmap="gray")

    # label the top row
    axes[0, 0].set_title("original image")
    plt.tight_layout()
