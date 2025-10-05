
import yaml
import abc
import equinox as eqx
import time
from pathlib import Path
import numpy as onp
from copy import deepcopy

from nnfe.ml import ML
from cardiax import FE_manager
from nnfe.sampling import Sampler
from nnfe.utils import Utilities
from nnfe.plotter import Plotter

class NNFE_base():
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
    TODO:
    - Make compatible with future refactor
    - Add more documentation
    - Add checkpointing
    - Add or extend logger
    """

    def __init__(self, param_file):
        """
        Initialization of NNFE object, 
        will have to look more into what is needed here
        Args:
            fe_handler (_type_): _description_
            ml (_type_): _description_
            sampler (_type_): _description_
        """

        self.fe_handler, self.ml, self.sampler, self.utility, self.plotter, self.nnfe_params = load_nnfe(param_file)
        self.problem = self.fe_handler.problem
        self.template_int_vars = deepcopy(self.problem.internal_vars)
        self.template_int_vars_surfaces = deepcopy(self.problem.internal_vars_surfaces)
        self.setup(self.nnfe_params)
        self.val_and_grads = eqx.filter_value_and_grad(self.loss_fct)

        return
    
    
    def setup(self, params):
        """
        Setup requirements for NNFE
        """

        inds = {k: i for i, k in enumerate(params["order"])}

        # Create function to set appropriate values with NN inputs
        if params["natural"]:
            if params["natural"]["internal"]:
                int_vars = self.problem.internal_vars

                def nnfe_set_int_vars(x):
                    for fe_key, fe_params in params["natural"]["internal"].items():
                        for var in fe_params:
                            int_vars[fe_key][var] = x[inds[var]] * self.template_int_vars[fe_key][var]
                    return int_vars

            if params["natural"]["surface"]:
                int_vars_surfaces = self.problem.internal_vars_surfaces

                def nnfe_set_int_vars_surf(x):
                    for fe_key, fe_params in params["natural"]["surface"].items():
                        for bc, vars in fe_params.items():
                            for var in vars:
                                int_vars_surfaces[fe_key][bc][var] = x[inds[var]] * self.template_int_vars_surfaces[fe_key][bc][var]
                    return int_vars_surfaces

                pass

        # Add stuff here for dirichlet bcs
        


        self.nnfe_set_int_vars = nnfe_set_int_vars
        self.nnfe_set_int_vars_surf = nnfe_set_int_vars_surf
        self.nnfe_set_bc_vars = lambda x: None

        return
    
    def train(self):
        """
        Main training loop for NNFE
        """

        self.plotter.plot_learning_rate(self.ml.lr_scheduler,
                           self.ml.optimizer_params["epochs"])

        loss_vals = []

        toc = time.time()
        for i in range(1, int(self.ml.optimizer_params["epochs"]) + 1):
            self.ml.network, self.ml.opt_state, train_loss = self.make_step(self.ml.network, self.sampler.X, self.ml.opt_state)
            loss_vals.append(train_loss)
            if i % self.utility.print == 0:
                print("Iteration: ", i, " Loss: ", train_loss)

            # Save model while training
            try:
                if i % self.utility.save == 0:
                    self.save()
                    print("Model saved at iteration: ", i)
                    print("Run in dir: ", self.utility.parent)
            except ZeroDivisionError:
                pass

        time_elapsed = time.time() - toc
        print("Training time: ")
        print(time_elapsed)
        print("Time per iter: ")
        print(time_elapsed / self.ml.optimizer_params["epochs"])

        # Save the model after training
        if self.utility.save:
            self.save(forced=True)
            onp.savetxt(self.utility.parent / "running.txt", onp.array([time_elapsed]))
        else:
            print("No save directory specified, skipping save")

        self.plotter.plot_loss(onp.hstack(loss_vals))

        return

    def test(self):
        """
        Test the accuracy of the model after training
        """

        print("Training Error")
        res_vecs = self.vcalc_res(self.ml.network, self.sampler.X)
        mean_vecs = (res_vecs**2).mean(axis=1)
        print("Average Residuals: ", mean_vecs)
        print("Max Residuals: ", res_vecs.max(axis=1))

        print("Testing Error")
        res_vecs = self.vcalc_res(self.ml.network, self.sampler.Y)
        mean_vecs = (res_vecs**2).mean(axis=1)
        print("Average Residuals: ", mean_vecs)
        print("Max Residuals: ", res_vecs.max(axis=1))
        return

    def save(self, forced=False):
        """
        Responsible for saving the model and other things
        May want to create a temp save and final save
        to split up training process just in case
        """

        try:
            path = self.utility.parent / self.utility.dirs_params["model_dir"]
            eqx.tree_serialise_leaves(path / "model.eqx", self.ml.network)
        except KeyError:
            print("No model_dir set in utility parameters, skipping save")
            return

        if forced:
            try:
                path = self.utility.parent / self.utility.dirs_params["model_dir"]
                eqx.tree_serialise_leaves(path / "model.eqx", self.ml.network)
            except KeyError:
                eqx.tree_serialise_leaves(self.utility.parent / "model.eqx", self.ml.network)
                print("Forced save if forgot to assign model_dir")


        return

    @abc.abstractmethod
    def calc_res(self):
        """
        Calculate the residuals for the PDE
        This is done by calling the FE_handler object
        and passing in the ML object
        """

        pass

    @abc.abstractmethod
    def loss_fct(self):
        """
        Calculate the residuals for the PDE
        This is done by calling the FE_handler object
        and passing in the ML object
        """

        pass

    @eqx.filter_jit
    def make_step(self, model, x, opt_state):
        diff_model, static_model = eqx.partition(model, self.ml.filter)
        loss_val, grads = self.val_and_grads(diff_model, static_model, x)
        updates, opt_state = self.ml.optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

def load_nnfe(param_file):
    """
    Load the objects that are required for NNFE object from
    a yaml template. This is required for reproducibility,
    tracking results, and maintaining organization. 

    Currently this is the FE_handler object and the ML object.
    *This will be adaptive.
    
    Args:
        param_file (str): Pathway to the yaml file

    Returns:
        (FE_Handler): The FE_Handler object
        (ML): The ML object
    """

    param_file = Path(param_file)
    parent = param_file.parent

    # Read in the parameter file
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    nnfe_params = params["NNFE"]

    # Load the Problem object defining the appropriate PDE
    utility = Utilities(params["Project_Utility"])

    fe_manager = FE_manager(parent / Path(params["fe_input_file"]),
                            savedir=utility.savedir)

    model_path = utility.parent / utility.dirs_params["model_dir"]
    ml = ML(Path(parent / params["ml_input_file"]), int(fe_manager.problem.num_total_dofs_all_vars),
            utility.key, savedir=utility.savedir, model_path=model_path / "model.eqx")

    sampler = Sampler(params["Sampler"])

    plotter = Plotter(params["Plotting"], utility)

    if params["Project_Utility"]["save"]:
        params["Project_Utility"]["save"] = False
        
        with open(utility.savedir / param_file.name, "w") as f:
            yaml.dump(params, f)
        
    else:
        input_dirs = None

    return fe_manager, ml, sampler, utility, plotter, nnfe_params
