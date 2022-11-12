# Real-time-control-of-an-octopus-arm-NNES
One of the robotic analogies of an octopus arm is soft continuum manipulators. Given their potential to perform complex tasks in unstructured environments as well as to operate safely around humans, with applications ranging from agriculture to surgery, there are various proof-of-concept soft continuum manipulators with distinct designs, mechanisms and actuations. However, current studies overwhelmingly employ simplified models like constant curvature approximations or steady-state assumptions, which do not fully exploit soft material properties like high nonlinearity and inherent compliancy. 

In this project, I obtain a real-time controller for a soft arm to reach random targets in its workspace. The control approach is not only tailored to the distributed and compliant mechanical system, but also fast enough for real-time applications, learning purposes, and scaling up to multi-arm systems.


## Model an octopus arm as a Cosserat rod.
Cosserat rod model can represent slender rods  with one dimensional representation in space and account for bend, twist, stretch, and shear; allowing all possible modes of deformation to be considered under a wide range of boundary conditions. It models the rod as a deformable curve with attached deformable vectors to characterize its orientation, and evolves the system using a set of coupled second-order, nonlinear partial differential equations. Here, I used [Elastica](https://www.cosseratrods.org).

## Activation model using an octopus-like muscular structure
To get closer to the robotic setup, the arm is coupled with an actuation model inspired by octopus musculature. For 2d motion, it’s sufficient to consider the transverse muscles that allow the arm to extend and the longitudinal muscles that allow the arm to bend. We define the muscle activation to be continuous along the arm and has value between 0 to 1. Based on the muscles’ offset to the centerline, the activation directly maps to the forces and the couples on the Cosserat rod.
[image:muscle_model.gif]


## Energy shaping control
Energy shaping control has been applied to the aforementioned arm by treating it as a hamiltonian control system [1,2]. The hamiltonian is the sum of kinetic energy and the potential energy, where the potential energy function is the key of the distributed control because (1) with the fixed boundary condition of the rod, the potential energy in terms of the configuration state can be described with the rod’s mechanical deformation. (2) The minima of the potential energy then represents the intrinsic deformations of the rod. Therefore, say if we want the arm to get to a specific configuration, we can define a desired potential energy function with minimum at the corresponding deformation. 

Based on this observation, energy shaping control law is designed so that it modifies the potential energy landscape with the minima at the desired static configuration. So for reaching a target, we simply need to solve for the desired configuration $\bar{q}$ (or the activation $\alpha(s)$ that corresponds to it)  so that energy shaping control law can bring the arm there dynamically.

[image:F15734B4-8255-4F36-A872-991B73E5D918-64869-0000D6142592849D/ES.gif]

However,  getting this activation is computationally expensive. We need to solve this optimization problem: 
[image:9BCCC9B7-81A9-4F49-A544-7C3075802F27-64869-0000D5E03F0B9378/NNESproblem.jpeg]
This means we not only have to solve for this distributed muscle activations, but also the stabilizing shape q bar that is unknown to us.

In the original paper, they apply an iterative method to solve this optimization problem. At every iteration, both $\bar{q}$ and $\alpha$ need to be updated, and this goes on until getting the solution. In addition, the procedure needs to done every time when there is a new target, which can be challenging for real-time applications. It also scales up when we need to consider more muscles or more arms (for an octopus robot). 

## Neural network energy shaping control (NNES)
For real-time applications and later training purposes, I build a fast solver to output activations to the energy shaping controller [3]. I replaced the iterative method with a mapping that can provide muscle activations given any targets in the workspace and the current activations. The mapping is trained with a loss function in terms of distance error and energy costs using gradient descent (unsupervised learning). The learned mapping can provide activation solutions to the energy shaping control. Together, this NNES controller can bring the arm tip to the target, with the shape determined by the activations.

[image:5D56F891-66C5-4ED0-85EE-32DE825C7906-64869-0000D85652129C73/NNESframework.jpeg]


## 200 times faster than the iterative method!
I implement ES with 100, 1000, and 10000 maximum number of iterations. While the ES performance depends on the allowed iterations, the result shows a clear trend of ES and NNES. Both algorithms are accurate in reaching targets with median distances less than 1% of the arm. However, NNES has the advantage of a significant reduction in solution time, where NN-ES outperforms ES by over 200x. That means, For 20000 evaluations, NNES needs only 20s, while ES will require more than 1 hour.

[image:FC377B4F-A909-47CD-A363-997054D21A31-64869-0000D8A96295C8D5/NNESperformance.jpeg]


