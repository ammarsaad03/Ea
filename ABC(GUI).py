import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import lhs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class OptimizationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Artificial Bee Colony Optimization")
        # Calculate the screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Set the window width and height
        window_width = 1070
        window_height =610
        
        # Calculate the position for the window to be centered on the screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        # Set the window's size and position
        self.master.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        

        # Function selection
        function_label = ttk.Label(master, text="Select Function:")
        function_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.function_chosen_var = tk.StringVar(master)
        self.function_combobox = ttk.Combobox(master, textvariable=self.function_chosen_var,
                                              values=("Sphere", "Rosenbrock", "Griewank"))
        
        self.function_combobox.grid(row=0, column=1,padx=10, pady=5,sticky="w")

        # Parameters
        parameters_label = ttk.Label(master, text="Parameters")
        parameters_label.grid(row=1, column=0, columnspan=2)
        # Create input fields
        self.label_num_bees = ttk.Label(master, text="Number of Bees:")
        self.label_num_bees.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.entry_num_bees = ttk.Entry(master)
        self.entry_num_bees.grid(row=2, column=1,padx=10, pady=5,sticky="w")

        self.label_dimensionality = ttk.Label(master, text="Dimensionality:")
        self.label_dimensionality.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.entry_dimensionality = ttk.Entry(master)
        self.entry_dimensionality.grid(row=3, column=1,padx=10, pady=5,sticky="w")

        self.label_search_range = ttk.Label(master, text="Search Range:")
        self.label_search_range.grid(row=4, column=0, padx=10, pady=5, sticky="e")
        self.entry_lower_bound = ttk.Entry(master)
        self.entry_lower_bound.grid(row=4, column=1,padx=10, pady=5,sticky="w")
        
        self.label_iterations = ttk.Label(master, text="Number of iterations:")
        self.label_iterations.grid(row=5, column=0, padx=10, pady=5, sticky="e")
        self.entry_iterations = ttk.Entry(master)
        self.entry_iterations.grid(row=5, column=1,padx=10, pady=5,sticky="w")
        
        # Create combobox for crossover method
        self.label_crossover = ttk.Label(master, text="Crossover Method:")
        self.label_crossover.grid(row=6, column=0, padx=10, pady=5, sticky="e")
        self.crossover_var = tk.StringVar(master)
        self.crossover_combobox = ttk.Combobox(master, textvariable=self.crossover_var)
        self.crossover_combobox['values'] = ["Blend Crossover", "Arithmetic Crossover"]
        self.crossover_combobox.grid(row=6, column=1,padx=10, pady=5,sticky="w")

        # Create combobox for mutation method
        self.label_mutation = ttk.Label(master, text="Mutation Method:")
        self.label_mutation.grid(row=7, column=0, padx=10, pady=5, sticky="e")
        self.mutation_var = tk.StringVar(master)
        self.mutation_combobox = ttk.Combobox(master, textvariable=self.mutation_var)
        self.mutation_combobox['values'] = ["Gaussian Mutation", "Uniform Mutation"]
        self.mutation_combobox.grid(row=7, column=1,padx=10, pady=5,sticky="w")

        # Create combobox for parent selection method
        self.label_parent_selection = ttk.Label(master, text="Parent Selection Method:")
        self.label_parent_selection.grid(row=8, column=0, padx=10, pady=5, sticky="e")
        self.parent_selection_var = tk.StringVar(master)
        self.parent_selection_combobox = ttk.Combobox(master, textvariable=self.parent_selection_var)
        self.parent_selection_combobox['values'] = ["FPS with windowing", "Tournment"]
        self.parent_selection_combobox.grid(row=8, column=1,padx=10, pady=5,sticky="w")

        self.label_initilization = ttk.Label(master, text="Initilization Method:")
        self.label_initilization.grid(row=9, column=0, padx=10, pady=5, sticky="e")
        self.initilization_var = tk.StringVar(master)
        self.initilization_combobox = ttk.Combobox(master, textvariable=self.initilization_var)
        self.initilization_combobox['values'] = ["Random", "Latin Hypercube"]
        self.initilization_combobox.grid(row=9, column=1,padx=10, pady=5,sticky="w")
        
        
        # Create buttons
        self.button_run_optimization = ttk.Button(master, text="Run Optimization", command=self.run_optimization)
        self.button_run_optimization.grid(row=10, column=0,columnspan=2,padx=10, pady=5,sticky='w')
        # Create Clear button
        self.button_clear = ttk.Button(master, text="Clear All", command=self.clear_plot)
        self.button_clear.grid(row=10, column=1,padx=10, pady=5,sticky='w')
        
        #Best cost label
        self.label_cost = ttk.Label(self.master, text="")
        self.label_cost.grid(row=11, column=0, padx=10, pady=5)
        
        # Matplotlib figure
        self.figure = plt.figure(figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Value")
        self.ax.set_title("Convergence Curve")
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().grid(row=0, column=2,rowspan=11, padx=10, pady=5)
    
    def clear_plot(self):
        self.label_cost.config(text="")
        self.ax.clear()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Value")
        # self.ax.set_title("Convergence Curve")
        self.canvas.draw()
        
        
    def run_optimization(self):
        num_bees = int(self.entry_num_bees.get())
        dimensionality = int(self.entry_dimensionality.get())
        bound = float(self.entry_lower_bound.get())
        crossover_method = self.crossover_var.get()
        mutation_method = self.mutation_var.get()
        function_chosen = self.function_chosen_var.get()
        num_iterations =int(self.entry_iterations.get())
        parent_selection_method = self.parent_selection_var.get()
        initilization=self.initilization_var.get()
        abc_without_mc = \
        ABCWithBlendCrossover(function_chosen, num_bees, num_iterations, dimensionality, bound, -bound,crossover_method,mutation_method,parent_selection_method,initialization=initilization)

        best_solution, best_cost, cost_history = abc_without_mc.optimize()
        self.label_cost.config(text="Best Cost: {:.4f}".format(best_cost))
        # Plot convergence curve
        title="Convergence Curve of "+function_chosen
        self.ax.set_title(title)
        text="Mutation:"+mutation_method+"      Crossover:"+crossover_method+"\nParent selection: "+parent_selection_method+"       initilization:"+initilization+"\nCost:"+str(best_cost)
        self.ax.plot(cost_history, label=text)
        self.ax.legend()
        self.canvas.draw()
               
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x):
    sum_term = sphere_function(x) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1
     
class ABCWithBlendCrossover:
    def __init__(self, objective_function, num_bees, num_iterations, dimensionality,\
                lower_bound, upper_bound,crossover_method,mutation_method,\
                parent_selection_method, alpha=0.5,tournament_size=5, replacement=False,\
                deterministic=True, probability=1.0,initialization="random"):
        # Get selected function
        function_name = objective_function
        if function_name == "Sphere":
            self.objective_function = sphere_function
        elif function_name == "Rosenbrock":
            self.objective_function = rosenbrock_function
        elif function_name == "Griewank":
            self.objective_function = griewank_function

        self.num_bees = num_bees
        self.num_iterations = num_iterations
        self.dimensionality = dimensionality
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.alpha = alpha  # Blend alpha parameter
        self.q =10
        self.best_solution = None
        self.best_fitness = np.inf
        self.cost_history = []
        self.tournament_size = tournament_size
        self.replacement = replacement
        self.deterministic = deterministic
        self.probability = probability
        self.crossover_method=crossover_method
        self.parent_selection_method=parent_selection_method
        self.mutation_method=mutation_method
        self.mutation_rate =0.3 
        self.initialization=initialization
        
    def initialize_population(self):
        if self.initialization=="Random":
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_bees, self.dimensionality))
        elif self.initialization == "Latin Hypercube":
            return self.generate_latin_hypercube_population()
        else:
            raise ValueError("Invalid initialization method")
    
    def generate_latin_hypercube_population(self):
        lhs_samples = lhs(self.dimensionality, samples=self.num_bees, criterion='maximin')
        scaled_lhs_samples = self.lower_bound + lhs_samples * (self.upper_bound - self.lower_bound)
        return np.array(scaled_lhs_samples)
    
    def evaluate_fitness(self, population):
        return np.array([self.objective_function(sol) for sol in population])

    def employed_bees_phase(self, population):
        new_population = population.copy()
        for i in range(len(population)):
            candidate_solution = population[i]
            trial_solution = self.generate_trial_solution(candidate_solution, population)
            candidate_fitness = self.objective_function(candidate_solution)
            trial_fitness = self.objective_function(trial_solution)
            
            if trial_fitness < candidate_fitness:
                new_population[i] = trial_solution
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_solution
                    self.best_fitness = trial_fitness
            
        return new_population

    
    def generate_trial_solution(self, candidate_solution, population):
        ###Fitness–Proportionate Selection  with windowing scaling
        if self.parent_selection_method =="FPS with windowing":
            full_fitness=[]
            for i in range(len(population)):
                candidate_solution = population[i]
               
                full_fitness.append(self.objective_function(candidate_solution))
            partner = self.parent_selection(population,full_fitness)
            
        elif self.parent_selection_method =="Tournment": 
            partner = self.select_parents_tournament(population)
            
        trial_solution = np.zeros_like(candidate_solution)
        if self.crossover_method=="Blend Crossover":
            for i in range(len(candidate_solution)):
                trial_solution[i] = candidate_solution[i] + self.alpha * (partner[i] - candidate_solution[i]) * np.random.uniform(-0.5, 0.5)
        elif self.crossover_method=="Arithmetic Crossover":
            for i in range(len(candidate_solution)):
                trial_solution[i] =self.crossover_arithmetic(candidate_solution[i],partner[i])
        return trial_solution
    def crossover_arithmetic(self,position1, position2):
        alpha = np.random.uniform(0, 1)  # Crossover coefficient
        offspring_position = alpha * position1 + (1 - alpha) * position2
        return offspring_position

    def select_partner(self, population):
        selected_index = np.random.choice(len(population))
        return population[selected_index]
    
    def select_parents_tournament(self, population):
        selected_indices = np.random.choice(len(population), size=self.tournament_size, replace=self.replacement)
        tournament_population = population[selected_indices]
        if self.deterministic:
            return min(tournament_population, key=self.objective_function)
        else:
            return np.random.choice(tournament_population, p=[self.probability] + [(1 - self.probability) / (self.tournament_size - 1)] * (self.tournament_size - 1))
    
    
    def windowing(self, fitness_values, worst_fitness_last_generation):
        # Adjust the raw fitness values using windowing
        adjusted_fitness_values = fitness_values - worst_fitness_last_generation
        # Ensure that adjusted fitness values are non-negative
        adjusted_fitness_values = np.maximum(adjusted_fitness_values, 0)
        return adjusted_fitness_values
    
    def parent_selection(self, population, fitness_values):
        # Calculate the worst fitness from the last generation
        worst_fitness_last_generation = np.min(fitness_values)
        # Apply windowing to adjust fitness values
        adjusted_fitness_values = self.windowing(fitness_values, worst_fitness_last_generation)
        # Calculate selection probabilities
        probabilities = adjusted_fitness_values / np.sum(adjusted_fitness_values)
        # Select parents based on probabilities
        selected_index = np.argmin(probabilities)  # select Min prob solution
        return population[selected_index]
    def onlooker_bees_phase(self, population):
        fitness_values = self.evaluate_fitness(population)
        probabilities = fitness_values / fitness_values.sum()
            
        new_population = []
        for i in range(self.num_bees):
            rand = np.random.rand()
            if rand < probabilities[i]:
                if self.parent_selection_method=="FPS with windowing":
                    new_solution = population[i] + np.random.uniform(-1, 1, size=self.dimensionality) * (population[i] - self.parent_selection(population,fitness_values))
                    new_population.append(self.mutation(new_solution,self.upper_bound))
                elif self.parent_selection_method=="Tournment":
                    new_solution = population[i] + np.random.uniform(-1, 1, size=self.dimensionality) * (population[i] - self.select_parents_tournament(population))
                    new_population.append(self.mutation(new_solution,self.upper_bound))
            else:
                new_population.append(population[i])
        return np.array(new_population)

    def scout_bees_phase(self, population):
        for i in range(len(population)):
            if np.random.rand() < 0.1:  # 10% probability for scouting
                population[i] = self.initialize_population()[0]
        return population
    
    
    def mutation(self,position,search_range):
        #Gausissian mutaion non-uniform
        if self.mutation_method=="Gaussian Mutation":
            mutated_position = position[:] - np.random.normal(0, self.mutation_rate, position.shape)# Gaussian mutation
        #uniform mutaion
        elif self.mutation_method=="Uniform Mutation":
            mutation_indices = np.random.rand(self.dimensionality) < self.mutation_rate
            mutated_position = position.copy()
            mutated_position[mutation_indices] = np.random.uniform(-search_range, search_range)
        return mutated_position
    def optimize(self):
        np.random.seed(1)
        population = self.initialize_population()
        for itr_indx in range(self.num_iterations):
            population = self.employed_bees_phase(population)
            population = self.onlooker_bees_phase(population)
            population = self.scout_bees_phase(population)
            self.cost_history.append(self.best_fitness)
        # # Plot the cost history
        # plt.plot(self.cost_history, label="ABC without mutation and crossover")

        # # plt.plot(cost_history_with_blend_crossover, label="ABC with mutation and crossover")
        # plt.xlabel('Iteration')
        # plt.ylabel('Cost')
        # plt.title('ABC')
        # plt.legend()
        # plt.show()

        return self.best_solution, self.best_fitness, self.cost_history

def main():
    root = tk.Tk()
    gui = OptimizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
