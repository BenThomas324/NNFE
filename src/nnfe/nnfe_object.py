
from nnfe import project_manager
import yaml
import jax
import jax.numpy as np
import equinox as eqx
import time
from pathlib import Path
import numpy as onp
from copy import deepcopy
from jax.typing import ArrayLike
import dataclasses

from nnfe.ml import MLManager
from cardiax import ProblemManager
from nnfe.sampling import Sampler
from nnfe.project_manager import ProjectManager
from nnfe.plotter import Plotter
from nnfe.nnfe_config import NNFEConfig, NNFEParamsConfig

class NNFE():
    """
    Base class for the NNFE object.
    There is a LOT of customization that can occur here, 
    making it more important than ever to create input files for 
    reproducibility and tracking.
    This class is the organizational structure that handles the subparts
    of the NNFE method. This includes:
    - Problem definition (problem)
    - Machine Learning model (ml)
    - Sampler for training/testing data (sampler)
    - Training loop
    - Testing loop
    - Saving/loading model
    """



    def __init__(self, 
                 problem_manager: ProblemManager,
                 ml_manager: MLManager,
                 sampler: Sampler,
                 plotter: Plotter,
                 project_manager: ProjectManager,
                 nnfe_params: NNFEParamsConfig,
                 config: NNFEConfig = None):
        """Initialise the NNFE solver from pre-built sub-components.

        Prefer :meth:`from_yaml` or :meth:`from_config` for typical use;
        this constructor is exposed for cases where the sub-components are
        constructed manually.

        Args:
            problem_manager: Wraps the FE problem definition (mesh, BCs,
                residual computation).
            ml_manager: Owns the neural network, optimizer, and training
                state.
            sampler: Provides training and testing point sets.
            plotter: Handles diagnostic plot generation.
            project_manager: Manages the run directory, keys, and saving.
            nnfe_params: Specifies which FE variables the network controls
                and their ordering in the output vector.
            config: The full :class:`~nnfe.nnfe_config.NNFEConfig` used to
                build this object; stored for later serialisation.
        """

        self.problem_manager = problem_manager
        self.ml = ml_manager
        self.sampler = sampler
        self.plotter = plotter
        self.project = project_manager
        self.nnfe_params = nnfe_params

        if config is not None:
            self.config = config
        if getattr(self.project, "trained_weights_path", None) is not None:
            self.ml.network = self.ml.load_network(self.ml.network, self.project.trained_weights_path)
        
        self.problem = self.problem_manager.problem
        self.dirichlet_dofs, self.dirichlet_vals = problem_manager.problem.get_boundary_data()
        self.template_int_vars = deepcopy(self.problem.internal_vars)
        self.template_int_vars_surfaces = deepcopy(self.problem.internal_vars_surfaces)
        self.template_bc_vals = deepcopy(self.problem.bc_vals)
        self.setup(nnfe_params)
        self.val_and_grads = eqx.filter_value_and_grad(self.loss_fct)
        self.vcalc_res = jax.vmap(self.calc_res, in_axes=(None, 0), out_axes=0)

        return
    
    @classmethod
    def from_yaml(cls, path):
        nnfe_config = NNFEConfig.from_yaml(path)
        return cls.from_config(nnfe_config)

    @classmethod
    def from_config(cls, config: NNFEConfig):

        cls.config = config

        # Load the Problem object defining the appropriate PDE
        project_manager = ProjectManager(config.project)

        problem_manager = ProblemManager(config.FE)

        model_path = project_manager.paths.get("model_dir", None)
        out_size = problem_manager.problem.num_total_dofs_all_vars

        new_ml_cfg = dataclasses.replace(config.ML, rng_key=project_manager.key)
        config = dataclasses.replace(config, ML=new_ml_cfg)

        ml_manager = MLManager(config.ML, out_size=out_size, 
                               model_path=model_path)

        sampler = Sampler(config.sampler)

        try:
            save_dir = project_manager.parent / config.project.extra_dirs["plot_dir"]
        except TypeError:
            save_dir = None

        plotter = Plotter(config.plotter, 
                          save_dir=save_dir)

        if project_manager.config.save and project_manager.parent is not None:
            # You can save it in the root run folder, or inside project_manager.paths.get("config_dir")
            cls.dump_config(cls, project_manager.paths["config_dir"], Path("resolved_nnfe_config.yaml"))
            if project_manager.print:
                print(f"Master configuration saved to: {project_manager.paths["config_dir"]}")

        return cls(problem_manager=problem_manager, 
                   ml_manager=ml_manager, 
                   sampler=sampler, 
                   plotter=plotter, 
                   project_manager=project_manager,
                   nnfe_params=config.NNFE,
                   config=config)

    def setup(self, config: NNFEParamsConfig):
        """Wire up the NNFE-specific internal-variable setters.

        Builds closures that map the network output vector to the FE problem's
        internal variables (volumetric and surface) and boundary condition
        values, using the orderings defined in *config*.  Also JIT-compiles
        the batch-drawing function and computes the integer batch size.

        Args:
            config: NNFE parameter config specifying which FE variables are
                controlled and in what order they appear in the output vector.
        """

        natural_inds = {k: i for i, k in enumerate(config.natural_order)}

        # Create function to set appropriate values with NN inputs
        if config.natural:
            if config.natural["internal"]:
                int_vars = self.problem.internal_vars

                def nnfe_set_int_vars(x):
                    for fe_key, fe_params in config.natural["internal"].items():
                        for var in fe_params:
                            int_vars[fe_key][var] = x[natural_inds[var]] * self.template_int_vars[fe_key][var]
                    return int_vars

            if config.natural["surface"]: 
                int_vars_surfaces = self.problem.internal_vars_surfaces

                def nnfe_set_int_vars_surf(x):
                    for fe_key, fe_params in config.natural["surface"].items():
                        for bc, vars in fe_params.items():
                            for var in vars:
                                int_vars_surfaces[fe_key][bc][var] = x[natural_inds[var]] * self.template_int_vars_surfaces[fe_key][bc][var]
                    return int_vars_surfaces

        # Currently making dirichlet only work with 1 control variable
        # That moves all values together, so nothing to do here
        # if config.essential:
        #     for fe_key, fe_params in config.essential.items():
        #         pass

        if config.essential_order == []:
            self.train_dirichlet = False
        else:
            self.train_dirichlet = True


        self.nnfe_set_int_vars = nnfe_set_int_vars
        self.nnfe_set_int_vars_surf = nnfe_set_int_vars_surf
        self.nnfe_set_bc_vars = lambda x: None

        # Setup batching
        self.batch_size = int(self.sampler.X.shape[0] * self.ml.batch_size)
        self.sampler.draw_batch = jax.jit(self.sampler.draw_batch, static_argnums=(1,))
        return
    
    def calc_res(self, model: eqx.Module, ctrl_vars: ArrayLike):
        """Compute the FE residual for a single control-variable sample.

        Evaluates the network to get the DoF vector, updates the FE internal
        variables accordingly, computes the global residual, and enforces
        Dirichlet boundary conditions via a lifting approach (strong
        enforcement).

        Args:
            model: Current Equinox network.
            ctrl_vars: 1-D array of control variables (network input) for
                this sample.

        Returns:
            Residual vector with Dirichlet rows replaced by
            ``dofs[dirichlet_dofs] - dirichlet_vals``.
        """
        # Evaluate model to get dofs
        dofs = model(ctrl_vars)
        # Set internal variables
        int_vars = self.nnfe_set_int_vars(ctrl_vars)
        int_vars_surfaces = self.nnfe_set_int_vars_surf(ctrl_vars)
        # Compute residual
        res_vec = self.problem.compute_residual_helper(dofs, int_vars, int_vars_surfaces)

        # Perform lift of dirichlet dofs
        res_vec = res_vec.at[self.dirichlet_dofs].set(dofs[self.dirichlet_dofs])
        if self.train_dirichlet:
            res_vec = res_vec.at[self.dirichlet_dofs].add(ctrl_vars[-1] * -self.template_bc_vals, unique_indices=True)
        else:
            res_vec = res_vec.at[self.dirichlet_dofs].add(-self.dirichlet_vals, unique_indices=True)
        return res_vec

    def loss_fct(self, diff_model: eqx.Module, static_model: eqx.Module, x: ArrayLike):
        """Batch loss: mean residual norm over a mini-batch.

        Recombines the differentiable and static model partitions, vectorises
        :meth:`calc_res` over the batch axis, and returns the mean
        :math:`\\ell_2` residual norm.

        Args:
            diff_model: Differentiable (trainable) partition of the network.
            static_model: Frozen (static) partition of the network.
            x: Batch of control-variable samples with shape
                ``(batch_size, n_ctrl_vars)``.

        Returns:
            Scalar mean residual norm for the batch.
        """
        model = eqx.combine(diff_model, static_model)
        vres = self.vcalc_res(model, x)
        return np.mean(np.linalg.norm(vres, axis=1))

    @eqx.filter_jit
    def make_step(self, model: eqx.Module, x: ArrayLike, opt_state: tuple):
        """Perform a single gradient-descent update step (JIT-compiled).

        Partitions the model into trainable and frozen parts, computes the
        loss and its gradient with respect to the trainable part, applies the
        Optax optimizer update, and returns the updated model and state.

        Args:
            model: Current Equinox network.
            x: Mini-batch of control-variable samples.
            opt_state: Current Optax optimizer state.

        Returns:
            Tuple of ``(updated_model, updated_opt_state, loss_value)``.
        """
        diff_model, static_model = eqx.partition(model, self.ml.filter)
        loss_val, grads = self.val_and_grads(diff_model, static_model, x)
        updates, opt_state = self.ml.optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    def train(self):
        """
        Main training loop for NNFE
        """

        self.plotter.plot_learning_rate(self.ml.lr_scheduler,
                           self.ml.epochs)

        rng_key = jax.random.key(self.project.key)
        rng_key, batch = self.sampler.draw_batch(rng_key, self.batch_size)

        toc = time.time()
        self.ml.network, self.ml.opt_state, train_loss = self.make_step(self.ml.network, batch, self.ml.opt_state)
        train_loss.block_until_ready()
        jit_time = time.time() - toc

        loss_vals = [train_loss]
        toc = time.time()
        for i in range(2, self.ml.epochs + 1):
            rng_key, batch = self.sampler.draw_batch(rng_key, self.batch_size)
            self.ml.network, self.ml.opt_state, train_loss = self.make_step(self.ml.network, batch, self.ml.opt_state)
            loss_vals.append(train_loss.item())
            if i % self.project.print == 0:
                print("Iteration: ", i, " Loss: ", train_loss)

            # Save model while training
            # Need to figure out how to stop saving later
            if self.project.save and i % self.project.save_progress == 0:
                self.save()
                print("Model saved at iteration: ", i)
                print("Run in dir: ", self.project.parent)

        time_elapsed = time.time() - toc
        print("Training time: ")
        print(time_elapsed)
        print("Time per iter: ")
        print(time_elapsed / self.ml.epochs)
        print("JIT compilation time: ")
        print(jit_time)

        # Save the model after training
        if self.project.save:
            self.save()
            onp.savetxt(self.project.parent / "running.txt", onp.array([time_elapsed]))
        else:
            print("No save directory specified, skipping save")

        self.plotter.plot_loss(onp.hstack(loss_vals))

        updated_project_config = dataclasses.replace(
            self.project.config, 
            trained_weights_path=f"./{self.project.paths['model_dir'] / 'model.eqx'}" if self.project.save else None,
            save=False
        )
        
        # Update the main config object
        self.config = dataclasses.replace(
            self.config, 
            project=updated_project_config
        )
        self.dump_config(self.project.paths["config_dir"], Path("resolved_nnfe_config.yaml"))

        return

    def test(self, x):
        """
        Test the accuracy of the model after training
        """

        nn_sol = self.evaluate(x)
        # Update internal vars
        self.problem_manager.problem.set_internal_vars(self.nnfe_set_int_vars(x))
        self.problem_manager.problem.set_internal_vars_surfaces(self.nnfe_set_int_vars_surf(x))
        # Update BCs
        if self.train_dirichlet:
            self.problem_manager.problem.set_bc_vals(x[-1] * self.template_bc_vals)
        self.problem_manager.solver.initial_guess = nn_sol
        fe_sol, info = self.problem_manager.solver.solve(max_iter=40)
        assert info[0]

        return nn_sol, fe_sol

    def save(self):
        """
        Responsible for saving the model and other things
        May want to create a temp save and final save
        to split up training process just in case
        """

        try:
            path = self.project.paths["model_dir"]
            eqx.tree_serialise_leaves(path / "model.eqx", self.ml.network)
        except KeyError:
            print("No model_dir set in utility parameters, skipping save")
            return
        
        return
    
    def evaluate(self, x: ArrayLike):
        """
        Evaluate the model at a given point x.
        Args:
            x: The input point to evaluate the model at.
        Returns:
            The output of the model at the input point x.
        """
        dofs = self.ml.network(x)
        if self.train_dirichlet:
            dofs = dofs.at[self.dirichlet_dofs].set(x[-1] * self.template_bc_vals)
        else:
            dofs = dofs.at[self.dirichlet_dofs].set(self.dirichlet_vals)
        return dofs
    
    def dump_config(self, save_dir: Path, filename: Path):
        """Dumps this specific manager's configuration to a YAML file."""
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename
        
        with open(file_path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self.config), f, sort_keys=False)