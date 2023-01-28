import os
import time as timer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from train_nnes import generate_train_data, make_rod, get_rod_parameter, generate_straight, tf_target_config, \
    tf_distribute_activation, tf_activation_to_muscle_force, tf_cal_dilatation, tf_cal_integral, tf_cal_energy, \
    tf_energy_difference, hull_pts
from workspace import plot_polygon_workspace

L0 = 0.2


def plot_UL_rod(arm_position, cost_UL, target, eps, model_dir, soln_distance=None):
    marker_size = np.linspace(1000, 10, arm_position.shape[1])
    fig, ax = plt.subplots(figsize=(24, 9))
    plot1 = plot_polygon_workspace(hull_pts, ax=ax)
    UL_color = 'mediumpurple'
    plot3, = ax.plot(arm_position[0, :] / L0, arm_position[1, :] / L0, UL_color, linewidth=4)
    for k in range(arm_position.shape[1]):
        ax.scatter(arm_position[0, k] / L0, arm_position[1, k] / L0, s=marker_size[k], color=UL_color,
                   alpha=0.3)
    plot4 = ax.scatter(target[0, 0], target[0, 1], marker='x', color="r", label='target', s=300, linewidths=7)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if soln_distance == None:
        plt.legend([plot1, plot3, plot4],
                   ['workspace',
                    'UL: error=%.2f%%, E=%.4f' % (cost_UL[1] * 100, cost_UL[2] / 300),
                    'target'], fontsize=28, bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_xlim(-0.55, 1.35)
    ax.set_ylim(-0.1, 1.2)
    ax.xaxis.set_ticks(np.arange(-0.5, 1.3, 0.5))
    ax.yaxis.set_ticks(np.arange(-0.0, 1.2, 0.5))
    plt.gca().set_aspect('equal', adjustable='box')

    ax.tick_params(which='both', width=2, labelsize=62)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    fig.tight_layout()
    fig.savefig(model_dir + '/result/%d.pdf' % eps, transparent=True)
    plt.close(fig)


def test_rod(init_activation, activation, rod_list, max_force_list, muscle_radius_ratio_list):
    distributed_activation = tf_distribute_activation(activation)
    sigma, kappa, muscle_forces, muscle_couples = tf_activation_to_muscle_force(rod_list, distributed_activation,
                                                                                max_force_list,
                                                                                muscle_radius_ratio_list)

    sigma1 = tf.concat([tf.zeros((1, 100), dtype=tf.float32), tf.zeros((1, 100), dtype=tf.float32),
                        tf.ones((1, 100), dtype=tf.float32)], axis=0)
    elongation = sigma + sigma1
    dilatation, voronoi_dilatation = tf_cal_dilatation(elongation)
    curvature = tf.multiply(kappa, 1 / voronoi_dilatation)
    shear_matrix, bend_matrix, position_collection, director_collection, dl, _, _, _, _, _, _, _ = rod_list

    tip, arm_position, arm_director = tf_cal_integral(
        dl, elongation, kappa, position_collection, director_collection, voronoi_dilatation)

    arm_position = np.array(arm_position)
    arm_position = arm_position.T
    _, shear_strain, bend_curvature = tf_cal_energy(sigma, curvature, shear_matrix, bend_matrix, dilatation,
                                                    voronoi_dilatation)
    _, diff_energy = tf_energy_difference(init_activation, activation)
    energy = diff_energy
    EIcurvature = bend_curvature.numpy()[:, 0, 0]
    GAstrain = shear_strain.numpy()[:, 1, 1]
    EAstrain = shear_strain.numpy()[:, 2, 2]
    arm_ES_on_UL, cost_ES_on_UL, J = None, None, None

    return EIcurvature, GAstrain, EAstrain, EIcurvature, GAstrain, EAstrain, arm_position, curvature, sigma, energy, arm_ES_on_UL, cost_ES_on_UL


def rod_summary(list2, target):
    EI, EIcurvature, curvature, GA, GAstrain, strain, EA, EAstrain, strain, arm, energy = list2
    tip = arm[:2, -1]
    distance = tf.norm(tip / L0 - target)
    J = energy.numpy() + 1000 / 2 * distance ** 2
    return [J, distance, energy]


def run_random(model_dir, n_sample):
    nn_new = tf.keras.models.load_model(model_dir + '/pretrained_model.h5')
    os.makedirs(model_dir + "/result", exist_ok=True)

    x = generate_train_data(n_sample)

    rod, env, max_force_list, muscle_radius_ratio_list = make_rod()
    rod_list = get_rod_parameter(rod)

    init_activation = generate_straight()[0]
    log_time = []
    log_accuracy = []
    log_energy = []
    log_arm = []
    for eps in range(len(x)):
        target = x[eps][:2][np.newaxis, :]

        train_state = tf.concat([target, init_activation[tf.newaxis, :]], axis=1)
        begin = timer.time()
        pred_activation_coeff = nn_new(train_state, training=False)
        end = timer.time()
        execution_time = end - begin

        activation = tf_target_config(pred_activation_coeff)

        EIcurvature, GAstrain, EAstrain, EI, GA, EA, arm_position, curvature, strain, energy, arm_ES_on_UL, cost_ES_on_UL = test_rod(
            tf.zeros_like(activation), activation, rod_list, max_force_list, muscle_radius_ratio_list)
        UL_list = [EI, EIcurvature, curvature, GA, GAstrain, strain, EA, EAstrain, strain, arm_position, energy]

        log_arm.append(arm_position)
        cost_UL = rod_summary(UL_list, target)

        plot_UL_rod(arm_position, cost_UL, target, eps, model_dir)

        log_time.append(execution_time)
        log_accuracy.append(cost_UL[1])
        log_energy.append(cost_UL[2])

    np.savez(model_dir + "/result/arm_shape", log_arm=log_arm)


if __name__ == '__main__':
    model_dir = "./"
    run_random(model_dir, n_sample=2)
