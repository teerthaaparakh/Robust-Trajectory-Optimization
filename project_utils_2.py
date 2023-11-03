import pydrake

import numpy as np
import os
import pydot
import sys

from manipulation.meshcat_cpp_utils import StartMeshcat
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors, AddShape
from manipulation.utils import FindResource
from manipulation import running_as_notebook

import pandas as pd

from pydrake.all import (Adder, AddMultibodyPlantSceneGraph, AngleAxis, AbstractValue,
                         BasicVector, BallRpyJoint, BaseField,
                         Box, CameraInfo, ClippingRange, CoulombFriction,
                         Demultiplexer, DiagramBuilder, DepthRange,
                         DepthImageToPointCloud, DepthRenderCamera,
                         FindResourceOrThrow, GeometryInstance, Integrator, InverseDynamicsController,
                         JacobianWrtVariable, LeafSystem,
                         MultibodyPositionToGeometryPose, ModelInstanceIndex,
                         MeshcatVisualizerCpp, MultibodyPlant, Parser, PassThrough, PrismaticJoint,
                         PiecewisePose, PiecewisePolynomial, Quaternion, RevoluteJoint,
                         RigidTransform, Rgba, RollPitchYaw, RotationMatrix, RgbdSensor,
                         SceneGraph, Simulator, TrajectorySource, SpatialInertia, Sphere,
                         StateInterpolatorWithDiscreteDerivative, UnitInertia
                        )
from pydrake.examples.manipulation_station import ManipulationStation
from manipulation.meshcat_cpp_utils import (
    StartMeshcat, MeshcatJointSliders)
from manipulation.scenarios import AddMultibodyTriad, SetColor
from pydrake.solvers.snopt import SnoptSolver


m_obj = 0.05
len_obj = 0.1
q0_init = [-np.pi/2, -np.pi/2, 0.0, -np.pi/2, 0, 0.0, 0]


def dataframe(trajectory, times, names):
    assert trajectory.rows() == len(names)
    values = trajectory.vector_values(times)
    data = {'t': times }
    for i in range(len(names)):
        data[names[i]] = values[i,:]
    return pd.DataFrame(data)


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path, 'iiwa')
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0_init[index])
            index += 1

    return iiwa

def AddBox(plant, shape, name, mass=1, mu=1, color=[.5, .5, .9, 1.0]):
    instance = plant.AddModelInstance(name)
    inertia = UnitInertia.SolidBox(shape.width(), shape.depth(),
                                    shape.height())
    body = plant.AddRigidBody(
        name, instance,
        SpatialInertia(mass=mass,
                       p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=inertia))
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            body, RigidTransform(),
            Box(shape.width() - 1e-5,
                shape.depth() - 1e-5,
                shape.height() - 1e-5),
            name,
            CoulombFriction(mu, mu))
        i = 0
        for x in [-shape.width() / 2.0, shape.width() / 2.0]:
            for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                    plant.RegisterCollisionGeometry(
                        body, RigidTransform([x, y, z]),
                        Sphere(radius=1e-7), f"contact_sphere{i}",
                        CoulombFriction(mu, mu))
                    i += 1

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)
    return instance, body


def MyManipulationStation(time_step=0.002, mu1 = 0.2, mu2 = 0.2):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    # Add Iiwa and slab
    iiwa = AddIiwa(plant)
#     iiwa.set_name('iiwa')
    slab_idx, slab_body = AddBox(plant, Box(0.5, 0.3, 0.05), 'slab',
                                 mass=0.05, mu=mu1, color=[.5, .5, .9, 1.0])
    X_7G = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 0.1])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa),
                     slab_body.body_frame(), X_7G)

    box_idx, box_body = AddBox(plant, Box(len_obj, len_obj, len_obj), 'box',
                                 mass=m_obj, mu=mu2, color=[.0, .5, .9, 1.0])

    # Add Camera
#     Parser(plant).AddModelFromFile(
#         FindResource("models/camera_box.sdf"), "camera0")

    # Finalize plant
    plant.Finalize()

    plant_context = plant.CreateDefaultContext()

    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('box'),
                           RigidTransform(RotationMatrix(), [0.0, 0.4, 2.5])) #0.114 + 0.05+len_obj/2

    num_iiwa_positions = plant.num_positions(iiwa)
    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(), "iiwa_position_command")

    # Export the iiwa "state" outputs.
    demux = builder.AddSystem(Demultiplexer(
        2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(iiwa), "iiwa_state_estimated")
    builder.ExportOutput(plant.get_generalized_acceleration_output_port(iiwa), "iiwa_generalized_acceleration")

    # controller_plant
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    controller_slab_idx, controller_slab_body = AddBox(controller_plant, Box(0.5, 0.3, 0.05), 'controller_slab',mass=0.05, mu=mu1, color=[.5, .5, .9, 1.0])
    controller_plant.WeldFrames(controller_plant.GetFrameByName("iiwa_link_7", controller_iiwa),controller_slab_body.body_frame(), X_7G)
    controller_plant.Finalize()

    # iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[100]*num_iiwa_positions,
            ki=[1]*num_iiwa_positions,
            kd=[20]*num_iiwa_positions,
            has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(
        plant.get_state_output_port(iiwa), iiwa_controller.get_input_port_estimated_state())

    # feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))

    # Use a PassThrough to make the port optional (it will provide zero values if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]*num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    # discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, time_step, suppress_initial_transient=True))
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(),
                    desired_state_from_position.get_input_port())

    # Export commanded torques.
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    # Cameras.
#     AddRgbdSensors(builder, plant, scene_graph)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                            "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                            "plant_continuous_state")

    # Build
    diagram = builder.Build()

    return diagram, plant, scene_graph, plant_context


class PositionController(LeafSystem):
    def __init__(self, plant, station, station_context):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self.station = station
        self.station_context = station_context
        self._iiwa = plant.GetModelInstanceByName("iiwa")
#         self._G = plant.GetBodyByName("slab").body_frame()
#         self._W = plant.world_frame()

#         self.w_G_port = self.DeclareVectorInputPort("omega_WG", BasicVector(3))
        self.q_dot_port = self.DeclareVectorInputPort("iiwa_velocity", BasicVector(7))
        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_position_out", BasicVector(7),
                                     self.CalcOutput)

#         self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7),
#                                      self.CalcOutput)

    def CalcOutput(self, context, output):
        q = self.q_port.Eval(context)
        q_dot = self.q_dot_port.Eval(context)
        
        q_curr = self.station.GetOutputPort("iiwa_position_measured").Eval(self.station_context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q_curr)
        # self._plant.SetVelocities(self._plant_context, self._iiwa, q_dot)
        output.SetFromVector(q)


def ExtendedStation(diagram, plant, meshcat, q_traj, q_dot_traj = None, diag_return = False):
    builder = DiagramBuilder()
    station = builder.AddSystem(diagram)
    temp_context = station.CreateDefaultContext()
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)

    initial_slab_pose = plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName('slab'))


    q_init = station.GetOutputPort("iiwa_position_measured").Eval(temp_context)
    
    controller = builder.AddSystem(PositionController(plant, station, temp_context))
    controller.set_name("PositionController")

    # q_traj = get_q_traj_(q_init)
    q_source = builder.AddSystem(TrajectorySource(q_traj))
    q_source.set_name("q_iiwa")

    q_dot_traj = q_traj.MakeDerivative()
    
    q_dot_source = builder.AddSystem(TrajectorySource(q_dot_traj))
    q_dot_source.set_name("q_dot_iiwa")

    builder.Connect(q_source.get_output_port(),
                    controller.GetInputPort('iiwa_position'))

    builder.Connect(q_dot_source.get_output_port(),
                    controller.GetInputPort('iiwa_velocity'))

    builder.Connect(controller.get_output_port(),
                    station.GetInputPort("iiwa_position"))

    meshcat.Delete()
    visualizer = MeshcatVisualizerCpp.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    diagram.set_name("traj_optim")

    simulator = Simulator(diagram)
    station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())

    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(sim_context)

    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName('box'),
                          RigidTransform(RotationMatrix(), [0.0, 0.4, 2.5]))
    if (diag_return):
        return simulator, plant_context, q_traj, q_dot_traj, diagram, station, station_context
    else:
        return simulator, plant_context, q_traj, q_dot_traj

#def run_setup(simulator, end_time):
#    if running_as_notebook:
#        simulator.set_target_realtime_rate(1.0)
#
#    simulator.AdvanceTo(end_time if running_as_notebook else 0.1)


##---------------------------------------------------------------------------------
## Algorithm Equations
##---------------------------------------------------------------------------------

def Iiwa_plant(mu1, time_step=0.002, with_box = False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    iiwa = AddIiwa(plant)

    slab_idx, slab_body = AddBox(plant, Box(0.5, 0.3, 0.05), 'slab',
                                 mass=0.1, mu=mu1, color=[.5, .5, .9, 1.0])
    X_7G = RigidTransform(RollPitchYaw(0, 0, 0.0), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa),
                     slab_body.body_frame(), X_7G)

    if with_box:
        box_idx, box_body = AddBox(plant, Box(len_obj, len_obj, len_obj), 'box',
                             mass=m_obj, mu=friction, color=[.0, .5, .9, 1.0])

    plant.Finalize()
    diagram = builder.Build()
    return diagram, plant
