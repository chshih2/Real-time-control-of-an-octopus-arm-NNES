import numpy as np
from sklearn.decomposition import PCA


def load_data(episode, n_episode):
    data = np.load("./ES_data/workspace_rod%03d.npz" % episode)

    rod = data['rod']
    shear = data['sigma']
    curvature = data['kappa']
    L0 = data['L0']
    target = data['target']
    activation = data['activation']
    return rod, shear, curvature, L0, target, activation


def find_basis_functions(activation, file_name):
    activation = np.array(activation)

    activation_mean = np.mean(activation, axis=0)
    activation_std = np.std(activation, axis=0)

    normalized_activation = (activation - activation_mean) / (activation_std + 1e-14)

    pca = PCA(n_components=11)

    pca_activation = pca.fit_transform(normalized_activation)

    np.savez(file_name,
             coefficients=pca_activation,
             basis_function=pca.components_,
             mean=activation_mean,
             std=activation_std)


if __name__ == '__main__':

    n_episode = 70

    activation = []
    for index in range(n_episode):
        each_activation = load_data(index, n_episode)[-1]
        activation.append(each_activation[0])
        activation.append(each_activation[1])
        activation.append(each_activation[2])

    find_basis_functions(activation, "activation")
