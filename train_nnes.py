import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras import Model, Input
import os, sys
import time as timer
from nnes_env import Environment
from workspace import generate_targets_inside_workspace, read_polygon_workspace


def read_pca_result(file):
    data = np.load(file)
    basis_function = data['basis_function']
    coefficients = data['coefficients']
    data_mean = data['mean']
    data_std = data['std']
    return basis_function, coefficients, data_mean, data_std


def make_rod():
    final_time = 2.5
    env = Environment(final_time)
    total_steps, systems = env.reset()
    max_force_list = []
    muscle_radius_ratio_list = []
    for i in range(len(env.muscle_layers)):
        max_force = env.muscle_layers[i].max_force
        max_force_list.append(make_tf_variable(max_force))
        muscle_radius_ratio = env.muscle_layers[i].muscle_radius_ratio
        muscle_radius_ratio_list.append(make_tf_variable(muscle_radius_ratio))

    return systems[0], env, max_force_list, muscle_radius_ratio_list


def make_tf_variable(x):
    x = tf.constant(x)
    x = tf.cast(x, tf.float32)
    return x


@tf.function
def get_rod_parameter(rod):
    shear_matrix = make_tf_variable(rod.shear_matrix)
    bend_matrix = make_tf_variable(rod.bend_matrix)
    position_collection = make_tf_variable(rod.position_collection)
    director_collection = make_tf_variable(rod.director_collection)
    dl = make_tf_variable(rod.lengths[0])
    radius = make_tf_variable(rod.radius)
    kappa = make_tf_variable(rod.kappa)
    tangents = make_tf_variable(rod.tangents)
    rest_lengths = make_tf_variable(rod.rest_lengths)
    rest_voronoi_lengths = make_tf_variable(rod.rest_voronoi_lengths)
    dilatation = make_tf_variable(rod.dilatation)
    voronoi_dilatation = make_tf_variable(rod.voronoi_dilatation)
    return [shear_matrix, bend_matrix, position_collection, director_collection, dl,
            radius, kappa, tangents, rest_lengths, rest_voronoi_lengths, dilatation, voronoi_dilatation]


def make_basis(k, file_name, eager=True):
    basis_function, coefficients, basis_mean, basis_std = read_pca_result(file_name + ".npz")
    if eager:
        basis = make_tf_variable(basis_function)
        activation_mean = make_tf_variable(np.mean(coefficients, axis=0))
        activation_std = make_tf_variable(np.std(coefficients, axis=0))
        basis_mean = make_tf_variable(basis_mean)
        basis_std = make_tf_variable(basis_std)
    else:
        basis = basis_function
        activation_mean = np.mean(coefficients, axis=0)
        activation_std = np.std(coefficients, axis=0)

    return basis, activation_mean, activation_std, basis_mean, basis_std


@tf.function
def tf_cal_dilatation(strain):
    dilatation = tf.sqrt(tf.einsum("ij,ij->j", strain, strain))
    voronoi_dilataion = (dilatation[:-1] + dilatation[1:]) / 2
    return dilatation, voronoi_dilataion


@tf.function
def tf_cal_energy(strain, curvature, shear_matrix, bend_matrix, dilatation, voronoi_dilatation):
    mm = tf.multiply(tf.multiply(shear_matrix, 1 / tf.multiply(dilatation, dilatation)), tf.multiply(strain, strain))
    mm2 = tf.transpose(mm)
    mm3 = tf.linalg.trace(mm2)
    mm4 = tf.math.reduce_sum(mm3)
    nn = tf.multiply(tf.multiply(bend_matrix, 1 / tf.multiply(voronoi_dilatation, voronoi_dilatation)),
                     tf.multiply(curvature, curvature))
    nn2 = tf.transpose(nn)
    nn3 = tf.linalg.trace(nn2)
    nn4 = tf.math.reduce_sum(nn3)
    return tf.add(mm4, nn4) / 2, mm2, nn2


def tf_cal_soln_diff_energy(strain, curvature, shear_matrix, bend_matrix):
    mm = tf.multiply(shear_matrix, tf.multiply(strain, strain))
    mm2 = tf.transpose(mm)
    mm3 = tf.linalg.trace(mm2)
    mm4 = tf.math.reduce_sum(mm3)
    nn = tf.multiply(bend_matrix,
                     tf.multiply(curvature, curvature))
    nn2 = tf.transpose(nn)
    nn3 = tf.linalg.trace(nn2)
    nn4 = tf.math.reduce_sum(nn3)
    return tf.add(mm4, nn4) / 2, mm2, nn2


@tf.function
def tf_next_position(elongation, dl, position, director):
    delta = tf.multiply(elongation, dl)
    position += tf.matmul(director, delta[:, tf.newaxis], transpose_a=True)[:, 0]
    return position


def tf_next_director(voronoi_dilatation, kappa, dl, director):
    element_rotation = tf.multiply(kappa, dl)
    angle = tf.norm(element_rotation, axis=0)
    if np.isclose(angle, 0):
        rotation_matrix = tf.eye(3, dtype=tf.float32)
    else:
        axis_tf = tf.multiply(element_rotation, 1 / angle)
        axis = axis_tf.numpy()
        k = 0.0  # tf.zeros_like(axis_tf[0])
        K = tf.constant([[k, -axis[2], axis[1]], [axis[2], k, -axis[0]], [-axis[1], axis[0], k]])
        K2 = tf.matmul(K, K)
        rotation_matrix = tf.multiply(tf.sin(angle), K) + tf.multiply(1 - tf.cos(angle), K2) + tf.eye(3)
    director = tf.matmul(rotation_matrix, director, transpose_a=True)
    return director


def tf_cal_integral(dl, elongation, kappa, position_collection, director_collection, voronoi_dilatation):
    director = director_collection[:, :, 0]
    position = position_collection[:, 0]
    arm_position = [position]
    arm_director = [director]
    for i in range(99):
        position = tf_next_position(elongation[:, i], dl, position, director)
        director = tf_next_director(voronoi_dilatation[i], kappa[:, i], dl, director)
        arm_position.append(position.numpy())
        arm_director.append(director.numpy())
    position = tf_next_position(elongation[:, 99], dl, position, director)
    arm_position.append(position.numpy())
    return position, arm_position, arm_director


@tf.function
def tf_difference_kernel(x):
    X = tf.concat([x[..., 0][:, tf.newaxis],
                   x[..., 1:] - x[..., :-1],
                   x[..., -1][:, tf.newaxis]],
                  axis=-1)
    return X


@tf.function
def tf_quadrature_kernel(x):
    X = tf.concat([x[..., 0][:, tf.newaxis],
                   x[..., 1:] + x[..., :-1],
                   x[..., -1][:, tf.newaxis]],
                  axis=1)
    return X / 2


@tf.function
def tf_muscle_to_strain(muscle_force, muscle_couple, shear_matrix, bend_matrix, dilatation, voronoi_dilatation):
    shear_matrix_diagonal = tf.linalg.diag_part(tf.transpose(shear_matrix))
    shear_matrix_diagonal = tf.transpose(shear_matrix_diagonal)
    bend_matrix_diagonal = tf.linalg.diag_part(tf.transpose(bend_matrix))
    bend_matrix_diagonal = tf.transpose(bend_matrix_diagonal)

    sigma = -muscle_force / (shear_matrix_diagonal / dilatation)
    kappa = -muscle_couple / (bend_matrix_diagonal / voronoi_dilatation ** 3)
    return sigma, kappa


@tf.function
def tf_activation_to_muscle_force(rod_list, distributed_activation, max_force_list, muscle_radius_ratio_list):
    shear_matrix, bend_matrix, position_collection, director_collection, dl, radius, kappa, tangents, rest_lengths, rest_voronoi_lengths, dilatation, voronoi_dilatation = rod_list
    force0 = tf.zeros((1, 100), dtype=tf.float32)
    sigma1 = tf.concat([tf.zeros((1, 100), dtype=tf.float32), tf.zeros((1, 100), dtype=tf.float32),
                        tf.ones((1, 100), dtype=tf.float32)], axis=0)

    for _ in range(3):
        force_list = []
        couple_list = []
        for i in range(len(max_force_list)):
            activation_force = tf.multiply(distributed_activation[i], max_force_list[i])
            internal_forces = tf.concat([force0, force0, activation_force[tf.newaxis, :]], axis=0)
            force_list.append(internal_forces)
            if i != 2:
                r_m = tf.multiply(muscle_radius_ratio_list[i], radius) / tf.sqrt(dilatation)
                activation_couple = [tf.linalg.cross(r_m[..., k], internal_forces[:, k])[:, tf.newaxis] for k in
                                     range(100)]
                activation_couple = tf.concat(activation_couple, axis=-1)
                internal_couples = tf_quadrature_kernel(activation_couple)[..., 1:-1]
                couple_list.append(internal_couples)
        total_internal_forces = tf.math.reduce_sum(force_list, axis=0)
        total_internal_couples = tf.math.reduce_sum(couple_list, axis=0)
        sigma, kappa = tf_muscle_to_strain(total_internal_forces, total_internal_couples, shear_matrix, bend_matrix,
                                           dilatation,
                                           voronoi_dilatation)
        elongation = sigma + sigma1
        dilatation, voronoi_dilatation = tf_cal_dilatation(elongation)

    return sigma, kappa, total_internal_forces, total_internal_couples


@tf.function
def tf_target_config(pred):
    pred1 = pred[:, :n_component]
    pred2 = pred[:, n_component:2 * n_component]
    pred3 = pred[:, 2 * n_component:]
    pred1 = pred1 * activation_longitudinal_std + activation_longitudinal_mean
    pred2 = pred2 * activation_longitudinal_std + activation_longitudinal_mean
    pred3 = pred3 * activation_transverse_std + activation_transverse_mean
    pred1 = tf.multiply(tf.matmul(pred1, activation_longitudinal_basis),
                        longitudinal_basis_std) + longitudinal_basis_mean
    pred2 = tf.multiply(tf.matmul(pred2, activation_longitudinal_basis),
                        longitudinal_basis_std) + longitudinal_basis_mean
    pred3 = tf.multiply(tf.matmul(pred3, activation_transverse_basis),
                        transverse_basis_std) + transverse_basis_mean
    activation1 = tf.clip_by_value(pred1, clip_value_min=1e-8, clip_value_max=1)
    activation2 = tf.clip_by_value(pred2, clip_value_min=1e-8, clip_value_max=1)
    activation3 = tf.clip_by_value(pred3, clip_value_min=1e-8, clip_value_max=1)
    activation = tf.concat([activation1, activation2, activation3], axis=0)
    return activation


def np_target_config(pred):
    if tf.is_tensor(pred):
        pred = pred.numpy()
    activation = pred_to_activation(pred)
    activation = np.clip(activation, a_min=1e-8, a_max=1)
    return activation


def pred_to_activation(pred):
    pred1 = pred[:, :n_component]
    pred2 = pred[:, n_component:2 * n_component]
    pred3 = pred[:, 2 * n_component:]
    pred1 = pred1 * activation_longitudinal_std_numpy + activation_longitudinal_mean_numpy
    pred2 = pred2 * activation_longitudinal_std_numpy + activation_longitudinal_mean_numpy
    pred3 = pred3 * activation_transverse_std_numpy + activation_transverse_mean_numpy
    pred1 = np.multiply(np.matmul(pred1, activation_longitudinal_basis_numpy),
                        longitudinal_basis_std_numpy) + longitudinal_basis_mean_numpy
    pred2 = np.multiply(np.matmul(pred2, activation_longitudinal_basis_numpy),
                        longitudinal_basis_std_numpy) + longitudinal_basis_mean_numpy
    pred3 = np.multiply(np.matmul(pred3, activation_transverse_basis_numpy),
                        transverse_basis_std_numpy) + transverse_basis_mean_numpy
    activation = np.concatenate([pred1, pred2, pred3], axis=0)
    return activation


def generate_straight(eager=True):
    if eager:
        pred1 = tf.multiply(- longitudinal_basis_mean, 1 / longitudinal_basis_std)
        pred1 = tf.matmul(pred1[tf.newaxis, :], tf.linalg.pinv(activation_longitudinal_basis))
        pred2 = tf.multiply(- transverse_basis_mean, 1 / transverse_basis_std)
        pred2 = tf.matmul(pred2[tf.newaxis, :], tf.linalg.pinv(activation_transverse_basis))
        pred1 = (pred1 - activation_longitudinal_mean) / activation_longitudinal_std
        pred2 = (pred2 - activation_transverse_mean) / activation_transverse_std
        pred = tf.concat([pred1, pred1, pred2], axis=-1)
    else:
        pred1 = np.multiply(- longitudinal_basis_mean_numpy, 1 / longitudinal_basis_std_numpy)
        pred1 = np.matmul(pred1[np.newaxis, :], np.linalg.pinv(activation_longitudinal_basis_numpy))
        pred2 = np.multiply(- transverse_basis_mean_numpy, 1 / transverse_basis_std_numpy)
        pred2 = np.matmul(pred2[np.newaxis, :], np.linalg.pinv(activation_transverse_basis_numpy))
        pred1 = (pred1 - activation_longitudinal_mean_numpy) / activation_longitudinal_std_numpy
        pred2 = (pred2 - activation_transverse_mean_numpy) / activation_transverse_std_numpy
        pred = np.concatenate([pred1, pred1, pred2], axis=-1)
    return pred


def activation2coeff(activations):
    pred0 = tf.multiply(activations[0] - longitudinal_basis_mean, 1 / longitudinal_basis_std)
    pred0 = tf.matmul(pred0[tf.newaxis, :], tf.linalg.pinv(activation_longitudinal_basis))
    pred1 = tf.multiply(activations[1] - longitudinal_basis_mean, 1 / longitudinal_basis_std)
    pred1 = tf.matmul(pred1[tf.newaxis, :], tf.linalg.pinv(activation_longitudinal_basis))
    pred2 = tf.multiply(activations[2] - transverse_basis_mean, 1 / transverse_basis_std)
    pred2 = tf.matmul(pred2[tf.newaxis, :], tf.linalg.pinv(activation_transverse_basis))
    pred0 = (pred0 - activation_longitudinal_mean) / activation_longitudinal_std
    pred1 = (pred1 - activation_longitudinal_mean) / activation_longitudinal_std
    pred2 = (pred2 - activation_transverse_mean) / activation_transverse_std
    pred = tf.concat([pred0, pred1, pred2], axis=-1)
    return pred


@tf.function
def tf_muscle_activation_energy(activation):
    energy = tf.linalg.trace(tf.matmul(activation, tf.transpose(activation)) / 100)
    return energy


@tf.function
def tf_energy_difference(init_activation, final_activation):
    energy = tf_muscle_activation_energy(final_activation)
    diff_activation = final_activation - init_activation
    # diff_activation = tf.math.maximum(diff_activation, tf.zeros_like(diff_activation))
    diff_energy = tf_muscle_activation_energy(diff_activation)
    return energy, diff_energy


def tf_cal_activation_energy(diff_activation, muscle_forces, muscle_couples, shear_matrix, bend_matrix):
    activation_squared = tf.multiply(diff_activation, diff_activation)

    force_coefficient = [tf.matmul(tf.matmul(muscle_forces[..., k][tf.newaxis, :], tf.linalg.inv(shear_matrix[..., k])),
                                   muscle_forces[..., k][:, tf.newaxis]) for k in range(100)]
    force_coefficient = tf.concat(force_coefficient, axis=1)
    force_coefficient = (force_coefficient[:, :-1] + force_coefficient[:, 1:]) / 2

    couple_coefficient = [
        tf.matmul(tf.matmul(muscle_couples[..., k][tf.newaxis, :], tf.linalg.inv(bend_matrix[..., k])),
                  muscle_couples[..., k][:, tf.newaxis]) for k in range(99)]
    couple_coefficient = tf.concat(couple_coefficient, axis=1)

    energy = tf.math.reduce_sum(0.5 * activation_squared * (force_coefficient + couple_coefficient))
    return energy


def tf_distribute_activation(activation):
    distributed_activation1 = tf_quadrature_kernel(activation[0][tf.newaxis, :])
    distributed_activation2 = tf_quadrature_kernel(activation[1][tf.newaxis, :])
    distributed_activation3 = tf_quadrature_kernel(activation[2][tf.newaxis, :])
    distributed_activation = tf.concat([distributed_activation1, distributed_activation2, distributed_activation3],
                                       axis=0)
    return distributed_activation


def cal_arm(activation, rod_list, max_force_list, muscle_radius_ratio_list):
    distributed_activation = tf_distribute_activation(activation)
    sigma, kappa, muscle_forces, muscle_couples = tf_activation_to_muscle_force(rod_list, distributed_activation,
                                                                                max_force_list,
                                                                                muscle_radius_ratio_list)

    sigma1 = tf.concat([tf.zeros((1, 100), dtype=tf.float32), tf.zeros((1, 100), dtype=tf.float32),
                        tf.ones((1, 100), dtype=tf.float32)], axis=0)
    elongation = sigma + sigma1
    dilatation, voronoi_dilatation = tf_cal_dilatation(elongation)
    shear_matrix, bend_matrix, position_collection, director_collection, dl, _, _, _, _, _, _, _ = rod_list
    tip, arm, _ = tf_cal_integral(dl, elongation, kappa, position_collection, director_collection, voronoi_dilatation)
    return tip, arm


def energy_shaping_loss(pred, x, max_force_list, muscle_radius_ratio_list, rod_list, w1, w2):
    target_pos = x[:, :2]
    init_activation = tf_target_config(x[:, 2:])
    pred_activation_coeff = pred

    activation = tf_target_config(pred_activation_coeff)
    tip, _ = cal_arm(activation, rod_list, max_force_list, muscle_radius_ratio_list)
    _, diff_energy = tf_energy_difference(init_activation, activation)

    distance = tf.norm(tip[:2] / L0 - target_pos)
    loss = w1 * diff_energy / 300.0 + w2 / 2 * distance ** 2
    return pred_activation_coeff, loss


# train function
def train(model, x, max_force, muscle_radius_ratio, rod_list, w1, w2):
    with tf.GradientTape() as g:
        predictions = model(x, training=True)
        activation, loss = loss_fn(predictions, x, max_force, muscle_radius_ratio, rod_list, w1, w2)
    gradients = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return activation


def generate_train_data(n_sample=1):
    x = []
    for _ in range(n_sample):
        x.append(generate_targets_inside_workspace(polygon))
    return x


@tf.function
def generate_init_activation():
    scale1, scale2 = 0.2, 1.0
    init_longitudinal_activation1 = tf.random.normal(shape=[n_component], mean=0.0,
                                                     stddev=1.0 * scale1)

    init_longitudinal_activation2 = tf.random.normal(shape=[n_component], mean=0.0,
                                                     stddev=1.0 * scale2)
    init_transverse_activation = tf.random.normal(shape=[n_component], mean=0.0,
                                                  stddev=1.0)
    init_activation = tf.concat(
        [init_longitudinal_activation1, init_longitudinal_activation2, init_transverse_activation], axis=0)
    return init_activation


def run_training(folder_dir, w1, w2):
    rod, env, max_force_list, muscle_radius_ratio_list = make_rod()

    rod_list = get_rod_parameter(rod)

    numBatches = 100
    from tqdm import tqdm
    for epoch in range(4000):
        t0 = timer.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        current_rate = 0.5 + 0.5 * 0.1 ** (epoch / 500)
        for index in tqdm(range(numBatches)):
            x = generate_train_data()
            batch_X = x[0][:2][np.newaxis, :]
            dice = np.random.rand()
            if dice > current_rate:
                init_activation = activation[0]
            elif dice < current_rate / 2:
                init_activation = generate_straight()[0]
            else:
                init_activation = generate_init_activation()

            train_state = tf.concat([batch_X, init_activation[tf.newaxis, :]], axis=1)
            activation = train(nn, train_state, max_force_list, muscle_radius_ratio_list, rod_list, w1, w2)
            lss = train_loss.result().numpy()

        with open(folder_dir + "/loss_learn_deform.txt", "a") as file1:
            file1.write("%.4f\n" % lss)
        nn.save(folder_dir + '/my_model.h5')

        T = round(timer.time() - t0, 2)
        sys.stdout.write(' time: %s test-loss: %s \n' % (T, lss))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
simulation_dir = "./"
n_component = 11
activation_range = tf.linspace(0, 1, 99)
activation_longitudinal_basis, activation_longitudinal_mean, activation_longitudinal_std, longitudinal_basis_mean, longitudinal_basis_std = make_basis(
    activation_range,
    simulation_dir + "/0531_11activation", eager=True)
activation_transverse_basis, activation_transverse_mean, activation_transverse_std, transverse_basis_mean, transverse_basis_std = make_basis(
    activation_range,
    simulation_dir + "/0531_11activation", eager=True)
activation_longitudinal_basis_numpy = activation_longitudinal_basis.numpy()
activation_longitudinal_mean_numpy = activation_longitudinal_mean.numpy()
activation_longitudinal_std_numpy = activation_longitudinal_std.numpy()
longitudinal_basis_mean_numpy = longitudinal_basis_mean.numpy()
longitudinal_basis_std_numpy = longitudinal_basis_std.numpy()
activation_transverse_basis_numpy = activation_transverse_basis.numpy()
activation_transverse_mean_numpy = activation_transverse_mean.numpy()
activation_transverse_std_numpy = activation_transverse_std.numpy()
transverse_basis_mean_numpy = transverse_basis_mean.numpy()
transverse_basis_std_numpy = transverse_basis_std.numpy()

scale = 1.0
L0 = 0.2 * scale
hull_pts, polygon = read_polygon_workspace(file_dir=simulation_dir)
if __name__ == '__main__':
    optimizer = "adam"
    w1 = 0.1
    w2 = 1000
    n_muscles = 3

    file_dir = "./"
    model_dir = "./"
    os.makedirs(model_dir, exist_ok=True)

    # define model
    X = Input(shape=(2 + n_component * n_muscles))
    nn_activation_fn = 'relu'
    H = Dense(128, activation=nn_activation_fn)(X)
    H = Dense(128, activation=nn_activation_fn)(H)
    H = Dense(128, activation=nn_activation_fn)(H)
    Y = Dense(n_component * n_muscles, activation='linear')(H)

    nn = Model(inputs=X, outputs=[Y])

    # loss and optimizer
    loss_fn = energy_shaping_loss
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    run_training(model_dir, w1, w2)
