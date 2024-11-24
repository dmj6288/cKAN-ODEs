using Random, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: Adam, mae, update!
using Flux
using Optimisers
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad

#this is a fix for an issue with an author's computer. Feel free to remove.
ENV["GKSwstype"] = "100"

# Directories
dir             = @__DIR__
dir             = dir*"/"
cd(dir)
fname           = "LV_MLP"
add_path        = "results_mlp/"
add_path_kan    = "results_kanode/"
figpath         = dir*add_path*"figs"
ckptpath        = dir*add_path*"checkpoints"

mkpath(figpath)
mkpath(ckptpath)

#define LV
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# define Lorenz
function lorenz!(du, u, p, t)
    # Extract state variables
    x, y, z = u
    # Standard parameters for Lorenz attractor
    σ = 10.0
    ρ = 28.0
    β = 8.0 / 3.0

    σ, ρ, β = p	

    # Lorenz system equations
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

#data generation parameters
timestep = 0.1
n_plot   = 1000
n_save   = 50
rng      = Random.default_rng()
Random.seed!(rng, 0)

tspan       = Float32.([0.0, 14])
tspan_train = Float32.([0.0, 3.5])

# Initial Condition

u0          = Float32.([1., 1., 1.])
p_          = Float32[10.0, 28.0, 8.0/3.0]

# Defining the ODE problem

prob        = ODEProblem(lorenz!, Float64.(u0), tspan, Float64.(p_))

# Integration

solution    = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index   = Int64(floor(length(solution.t)*tspan_train[2]/tspan[2]))                # number of training time points
t           = Float32.(solution.t)                                                    # full dataset
t_train     = t[1:end_index]                                                          # training cut
X           = Array(solution)
Xn          = deepcopy(X) 


# Define MLP 
###As in KANODE code, the layers can be modified here to recreate the testing in section A2 of the manuscript

# Defining a vanilla MLP with tanh() activation and dense layers

MLP         = Lux.Chain(Lux.Dense(2 => 50, tanh), Lux.Dense(50 => 2)) #like in https://github.com/RajDandekar/MSML21_BayesianNODE/blob/main/BayesiaNODE_SGLD_LV.jl

## **also note here that the KANODE and MLP-NODE codes use different packages.
## **so if the Dense command fails, make sure to use a different REPL (i.e. one that you did not previously run the KANODE code on)

p_, sT_     = Lux.setup(rng, MLP)                      # Initializes the parameters and the states of the MLP
pM_data     = getdata(ComponentArray(p_))              # Converts model parameters into structured array format
pM_axis     = getaxes(ComponentArray(p_))              # Retrieves axes or dimensions associated with the structured array
p           = Float32.(deepcopy(pM_data)./1e5)         # Converts the parameters to float and scales them by 1e5

# Define Neural ODE with MLP

train_node      = NeuralODE(MLP, tspan_train, Tsit5(), saveat = t_train); #neural ode
train_node_test = NeuralODE(MLP, tspan,       Tsit5(), saveat = t);       #neural ode for test part

# Function for forward prediction using train NODE

function predict(p)
    Array(train_node(u0, p, sT_)[1])
end

# Function for training loss calculation

function loss(p)
    mean(abs2, Xn[:, 1:end_index].- predict(ComponentArray(p,pM_axis)))
end

# Function for forward prediction using test NODE

function predict_test(p)
    Array(train_node_test(u0, p, sT_)[1])
end

# Function for test loss calculation using test node

function loss_test(p)
    mean(abs2, Xn .- predict_test(ComponentArray(p,pM_axis)))
end

# TRAINING

du = Float32.([0.0; 0.0; 0.0])
#p = deepcopy(train_node.p)

print("parameter size:")
print(length(p))

opt       = Adam(1e-2)
l         = []
l_test    = []

p_list    = []
N_iter    = 1e5
i_current = 1

function plotter(l, p_list, epoch)

    l_min   = minimum(l)                                   # Find minimum in batch of losses
    idx_min = findfirst(x -> x == l_min, l)                # Find index of minimum batch
    plt     = Plots.plot(l, yaxis=:log, label="train")     # Define a plot object for log transformed training loss

    plot!(l_test,           yaxis=:log, label="test")      # Plot test loss
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss.png"))
    print("minimum train loss: ")
    print(minimum(l))
    print("minimum test loss: ")
    print(minimum(l_test))

    p_opt = p_list[idx_min]
    train_node_ = NeuralODE(MLP, tspan, Tsit5(), saveat = timestep); #neural ode
    pred_sol_true = solution
    p_curr = p_list[end]
    pred_sol_kan = train_node_(u0, ComponentArray(p_curr,pM_axis), sT_)[1]
    plt=scatter(pred_sol_true, alpha = 0.75)
    plot!(pred_sol_kan)
    vline!([3.5], color=:black, label = "train/test split")
    xlabel!("Time [s]")
    ylabel!("x, y")
    png(plt, string(figpath, "/training/results.png"))

    #packaging various quantities, then saving them to a .mat file
    p_list_ = zeros(size(p_list,1),size(p_list[1],1),size(p_list[1],2))
    for j = 1:size(p_list,1)
        p_list_[j,:,:] = p_list[j]
    end
    l_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_[j] = l[j]
    end
    
    l_test_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_test_[j] = l_test[j]
    end
    pred_sol_kan = train_node_(u0, ComponentArray(p_opt,pM_axis), sT_)[1]
    file = matopen(dir*add_path*"checkpoints/"*fname*"_results_MLP.mat", "w")
    write(file, "p_list", p_list_)
    write(file, "loss", l_)
    write(file, "loss_test", l_test_)
    write(file, "kan_pred_t", pred_sol_kan.t)
    write(file, "kan_pred_u1", reduce(hcat,pred_sol_kan.u)'[:, 1])
    write(file, "kan_pred_u2", reduce(hcat,pred_sol_kan.u)'[:, 2])
    close(file)

end

iters=tqdm(1:N_iter-i_current)
for i in iters
   global i_current
    
   # gradient computation
   grad = Zgrad(loss, p)[1] 

   #model update
   update!(opt, p, grad)

   #loss metrics
   loss_curr=deepcopy(loss(p))
   loss_curr_test=deepcopy(loss_test(p))
   append!(l, [loss_curr])
   append!(l_test, [loss_curr_test])
   append!(p_list, [deepcopy(p)])
   set_description(iters, string("Loss:", loss_curr))
   i_current = i_current + 1


   if i%n_plot==0
       plotter(l, p_list, i)
   end

    
end







