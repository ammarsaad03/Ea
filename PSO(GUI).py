import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyDOE2 import lhs
class PSO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Swarm Optimization")
        
        # Calculate the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set the window width and height
        window_width = 1070
        window_height =610
        
        # Calculate the position for the window to be centered on the screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        # Set the window's size and position
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        

        self.function_var = tk.StringVar(value="Rosenbrock")

        # Function selection
        function_label = ttk.Label(root, text="Select Function:")
        function_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.function_combobox = ttk.Combobox(root, textvariable=self.function_var, values=("Sphere", "Rosenbrock", "Griewank"))
        self.function_combobox.grid(row=0, column=1, padx=10, pady=5)

        # Parameters
        parameters_label = ttk.Label(root, text="Parameters:")
        parameters_label.grid(row=1, column=0, columnspan=2)

        num_particles_label = ttk.Label(root, text="Number of Particles:")
        num_particles_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.num_particles_entry = ttk.Entry(root)
        self.num_particles_entry.grid(row=2, column=1, padx=10, pady=5)

        num_dimensions_label = ttk.Label(root, text="Number of Dimensions:")
        num_dimensions_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.num_dimensions_entry = ttk.Entry(root)
        self.num_dimensions_entry.grid(row=3, column=1, padx=10, pady=5)

        search_range_label = ttk.Label(root, text="Search Range:")
        search_range_label.grid(row=4, column=0, padx=10, pady=5, sticky="e")
        self.search_range_entry = ttk.Entry(root)
        self.search_range_entry.grid(row=4, column=1, padx=10, pady=5)
        
        max_iter_label = ttk.Label(root, text="Max Iterations:")
        max_iter_label.grid(row=5, column=0, padx=10, pady=5, sticky="e")
        self.max_iter_entry = ttk.Entry(root)
        self.max_iter_entry.grid(row=5, column=1, padx=10, pady=5)
        
        
        # Create combobox for crossover method
        self.label_crossover = ttk.Label(root, text="Crossover Method:")
        self.label_crossover.grid(row=6, column=0, padx=10, pady=5, sticky="e")
        self.crossover_var = tk.StringVar(root)
        self.crossover_combobox = ttk.Combobox(root, textvariable=self.crossover_var)
        self.crossover_combobox['values'] = ["Uniform Crossover", "Arithmetic Crossover"]
        self.crossover_combobox.grid(row=6, column=1,padx=10, pady=5,sticky="w")

        # Create combobox for mutation method
        self.label_mutation = ttk.Label(root, text="Mutation Method:")
        self.label_mutation.grid(row=7, column=0, padx=10, pady=5, sticky="e")
        self.mutation_var = tk.StringVar(root)
        self.mutation_combobox = ttk.Combobox(root, textvariable=self.mutation_var)
        self.mutation_combobox['values'] = ["Gaussian", "polynomial"]
        self.mutation_combobox.grid(row=7, column=1,padx=10, pady=5,sticky="w")

        # Create combobox for parent selection method
        self.label_parent_selection = ttk.Label(root, text="Parent Selection Method:")
        self.label_parent_selection.grid(row=8, column=0, padx=10, pady=5, sticky="e")
        self.parent_selection_var = tk.StringVar(root)
        self.parent_selection_combobox = ttk.Combobox(root, textvariable=self.parent_selection_var)
        self.parent_selection_combobox['values'] = ["Tournment","Exponential Rank"]
        self.parent_selection_combobox.grid(row=8, column=1,padx=10, pady=5,sticky="w")

        self.label_initilization = ttk.Label(root, text="Initilization Method:")
        self.label_initilization.grid(row=9, column=0, padx=10, pady=5, sticky="e")
        self.initilization_var = tk.StringVar(root)
        self.initilization_combobox = ttk.Combobox(root, textvariable=self.initilization_var)
        self.initilization_combobox['values'] = ["Random", "Latin Hypercube"]
        self.initilization_combobox.grid(row=9, column=1,padx=10, pady=5,sticky="w")
        
        
        
        # Run PSO button
        run_button = ttk.Button(root, text="Run PSO", command=self.run_pso)
        # run_button.grid(row=10, column=0, columnspan=2, pady=10)
        run_button.grid(row=10, column=0,columnspan=2,padx=10, pady=5,sticky='w')
        # Create Clear button
        # Create Clear button
        self.button_clear = ttk.Button(self.root, text="Clear All", command=self.clear_plot)
        self.button_clear.grid(row=10, column=1,padx=10, pady=5,sticky='w')
        
        #Best cost label
        self.label_cost = ttk.Label(self.root, text="")
        self.label_cost.grid(row=11, column=0, padx=10, pady=5)

        # Matplotlib figure
        self.figure = plt.figure(figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Value")
        self.ax.set_title("Convergence Curve")
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=13, padx=10, pady=5)
    
    
    def clear_plot(self):
        self.label_cost.config(text="")
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Value")
        self.canvas.draw()
    def run_pso(self):
        np.random.seed(1)
        try:
            # Get parameters
            num_particles = int(self.num_particles_entry.get())
            num_dimensions = int(self.num_dimensions_entry.get())
            max_iter = int(self.max_iter_entry.get())
            search_range = float(self.search_range_entry.get())
            crossover_method = self.crossover_var.get()
            mutation_method = self.mutation_var.get()
            # function_chosen = self.function_chosen_var.get()
            parent_selection_method = self.parent_selection_var.get()
            initilization=self.initilization_var.get()
            
            # Get selected function
            function_name = self.function_var.get()
            if function_name == "Sphere":
                objective_function = sphere_function
            elif function_name == "Rosenbrock":
                objective_function = rosenbrock_function
            elif function_name == "Griewank":
                objective_function = griewank_function

            # Run PSO
            function_chosen = objective_function
            
            
            # Run PSO with mutation and crossover
            best_position_with_mutation_and_crossover, best_value_with_mutation_and_crossover, convergence_curve_with_mutation_and_crossover =\
                pso_with_mutation_crossover(function_chosen, num_particles, num_dimensions,\
                    max_iter, search_range,mutation=mutation_method,crossover=crossover_method,\
                    Initilization=initilization,parent_selection=parent_selection_method)
            print("this isisissisisisi,",parent_selection_method)
            self.label_cost.config(text="Best Cost: {:.4f}".format(best_value_with_mutation_and_crossover))
            # Plot convergence curve
            title="Convergence Curve of "+function_chosen.__name__
            self.ax.set_title(title)
            text="Mutation:"+mutation_method+"       Crossover:"+crossover_method+"\nParent selection: "+parent_selection_method+"       initilization:"+initilization+"\nCost:"+str(best_value_with_mutation_and_crossover)
            self.ax.plot(convergence_curve_with_mutation_and_crossover, label=text)
            self.ax.legend()
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values for parameters.")

            
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x):
    sum_term = sphere_function(x) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1


def crossover_arithmetic(position1, position2):
    alpha = np.random.uniform(0, 1)  # Crossover coefficient
    offspring_position = alpha * position1 + (1 - alpha) * position2
    return offspring_position
def crossover_uniform(position1, position2):
    mask = np.random.randint(0, 2, size=position1.shape, dtype=bool)
    offspring_position = np.where(mask, position1, position2)
    return offspring_position

def mutationn(position, search_range):
    mutated_position = position[:] - np.random.normal(0, 0.1, position.shape)  # Gaussian mutation
    return mutated_position  # Clip to ensure positions are within search range
def mutation_polynomial(position, search_range):
    eta_m = 20
    low = -search_range
    high = search_range
    y = np.random.uniform(low, high, size=position.shape)
    delta1 = (y - low) / (high - low)
    delta2 = (high - y) / (high - low)
    mut_pow = 1.0 / (eta_m + 1.0)
    deltaq = np.zeros(position.shape)
    xy = 1.0 - delta1
    val = 2.0 * np.random.random(position.shape)
    xy[val <= 1.0] = np.power(val[val <= 1.0], mut_pow)
    deltaq = xy - 1.0
    position += deltaq * (high - low)
    position = constraint_handler(position, search_range)
    return position

def generate_latin_hypercube_population(num_particles, num_dimensions,search_range):
    lhs_samples = lhs(num_dimensions, samples=num_particles, criterion='maximin')
    scaled_lhs_samples = -search_range + lhs_samples * (search_range - -search_range)
    return np.array(scaled_lhs_samples)
def select_parents_tournament(particles_position, objective_function):
    # Number of participants in the tournament
    tournament_size = 2
    # Randomly select participants for the tournament
    participants_indices = np.random.choice(range(len(particles_position)), size=tournament_size, replace=False)
    # Evaluate the fitness of each participant
    participants_fitness = [objective_function(particles_position[i]) for i in participants_indices]
    # Select the index of the participant with the highest fitness
    winner_index = participants_indices[np.argmin(participants_fitness)]
    return winner_index
def pso_with_mutation_crossover(objective_function, num_particles, num_dimensions,\
        max_iter, search_range, mutation_rate=0.3,crossover_rate=0.9,\
        w=0.4, c1=1, c2=2,crossover="Uniform Crossover",\
        mutation="random",Initilization="Random",parent_selection="Tournment"):
    
    if (objective_function.__name__=="griewank_function"):
        num_particles=num_dimensions    
    
    mutation_method=mutation
    crossover_method=crossover
    parent_selection_method=parent_selection
    Initilization_method=Initilization
    xMin, xMax = -search_range, search_range
    vMin, vMax = -0.2 * (xMax - xMin), 0.2 * (xMax - xMin)
    this=0
    if Initilization_method=="Random":
        this=  np.random.uniform(-search_range, search_range, size=(num_particles, num_dimensions))
    elif Initilization_method == "Latin Hypercube":
        this= generate_latin_hypercube_population(num_particles, num_dimensions,search_range)
    particles_position =this
    # particles_position = np.random.uniform(xMin, xMax, (num_particles,num_dimensions))
    particles_velocity = np.random.uniform(vMin, vMax, (num_particles,num_dimensions))
    cost = np.zeros(num_particles)
    cost[:] = objective_function(particles_position[:])
    personal_best_position = np.copy(particles_position)
    personal_best_value = np.copy(cost)
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]
    global_best_value = personal_best_value[global_best_index]
    convergence_curve = np.zeros(max_iter)
    
    
    
    for iter_idx in range(max_iter):
        for i in range(num_particles):
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            cognitive_component = c1 * r1 * (personal_best_position[i] - particles_position[i])
            social_component = c2 * r2 * (global_best_position - particles_position[i])
            inertia_component = w * particles_velocity[i]
            particles_velocity[i] = inertia_component + cognitive_component + social_component
            particles_position[i] += particles_velocity[i]
            particles_velocity[i] = constraint_handler(particles_velocity[i], vMax)
            particles_position[i] = constraint_handler(particles_position[i], search_range)

            if np.random.rand() < mutation_rate:
                if mutation_method=="Gaussian":  
                    mutated_cost=-np.inf
                    mutated_position = mutationn(particles_position[i], search_range)
                    mutated_cost = objective_function(mutated_position)
                elif mutation_method=="polynomial":
                    mutated_position = mutation_polynomial(particles_position[i], search_range)
                    mutated_cost = objective_function(mutated_position)
                if mutated_cost < cost[i]:
                    particles_position[i] = mutated_position
                    cost[i] = mutated_cost

            if np.random.rand() < crossover_rate:
                partner_index = np.random.choice(num_particles)
                if parent_selection_method == "Exponential Rank":
                    # Calculate normalization constant
                    c = (1 - np.exp(-num_particles)) / (1 - np.exp(-1))
                    print("c: ",c)
                    # Calculate selection probabilities based on rank
                    selection_probabilities = np.exp(-np.arange(num_particles)) / c

                    print("ss: ",selection_probabilities)
                    # Select one parent based on selection probabilities
                    partner_index = np.random.choice(np.arange(1, num_particles + 1), p=selection_probabilities)
                    print("index: ",partner_index)
                elif parent_selection_method== "Tournment":
                    # Tournment parent selection
                    partner_index = select_parents_tournament(particles_position,objective_function)

                
                if crossover_method =="Arithmetic Crossover":
                    offspring_position= crossover_arithmetic(particles_position[i], particles_position[partner_index])
                elif  crossover_method=="Uniform Crossover" :
                    offspring_position= crossover_uniform(particles_position[i], particles_position[partner_index])
                
                offspring_cost = objective_function(offspring_position)
                
                if offspring_cost < cost[i]:
                    particles_position[i] = offspring_position
                    cost[i] = offspring_cost

            if cost[i] < personal_best_value[i]:
                personal_best_position[i] = particles_position[i].copy()
                personal_best_value[i] = cost[i]
                if personal_best_value[i] < global_best_value:
                    global_best_position = personal_best_position[i].copy()
                    global_best_value = personal_best_value[i]

        convergence_curve[iter_idx] = global_best_value

    return global_best_position, global_best_value, convergence_curve

def constraint_handler(position, search_range):
    return np.clip(position, -search_range, search_range)


if __name__ == "__main__":
    root = tk.Tk()
    app = PSO_GUI(root)
    root.mainloop()


