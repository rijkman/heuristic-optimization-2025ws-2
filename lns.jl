#!/usr/bin/env julia
include("solution_iter_delta.jl")
include("greedy.jl")
include("local_search_delta.jl")
include("instance_parsing.jl")
include("greedy_insertion.jl")

# High level:
# Initial solution
# while termination not met:
#     Destroy part of the solution
#     Repair the solution
#     Apply local improvement
#     Accept or reject solution
# return best solution found

using Random

function LNS(
    instance::PDPInstance;
    init_solution::PDPSolutionVector,
    max_time=100,
    α=0.1 # percentage of requests to remove
) 
    # termination criterium
    start_time = time()
    master_solution = init_solution
    @info "init" objective_value(instance, master_solution)
    while time() - start_time < max_time

        # destruction operator: remove α% request pairs
        k = floor(α * sum([length(x) for x in master_solution]))
        hyp_solution = deepcopy(master_solution)
        for _ in 1:k
            # random selection of removed requests
            v_index = rand([k for k in 1:instance.n_vehicles if length(hyp_solution[k]) > 0])
            r_index = rand(1:length(hyp_solution[v_index]))

            d_req = hyp_solution[v_index][r_index]
            pickup_idx, dropoff_idx, is_pickup, _ = instance.requests[d_req]

            if !(pickup_idx in hyp_solution[v_index] && dropoff_idx in hyp_solution[v_index])
                continue
            end 

            deleteat!(hyp_solution[v_index], findfirst(==(pickup_idx), hyp_solution[v_index]))
            deleteat!(hyp_solution[v_index], findfirst(==(dropoff_idx),hyp_solution[v_index]))
        end

        # greedy insertion heuristic
        hyp_solution = greedy_insertion(
            instance,
            hyp_solution,
            true
        )

        # local search
        _, _, hyp_solution, best_score = delta_local_search(
            instance;
            solution=hyp_solution,
            score=objective_value(instance, hyp_solution),
            neighborhood_func=delta_neighbor_in_switch_location,
            step_func=delta_step_best,
            stop_time=1.0,
            verbose=false
        )

        @info "Best Score: " best_score
        if best_score < objective_value(instance, master_solution)
            master_solution = deepcopy(hyp_solution)
        end
    end

    return master_solution
end

instance = read_PDPInstance("instances/100/competition/", "instance61_nreq100_nveh2_gamma91.txt")

(iter_score, iter_n, best_solution, best_score) = greedy_heuristic_one_extend(instance; verbose=false)
@info "results:" is_feasible(instance, best_solution) objective_value(instance, best_solution)
sol = LNS(instance; init_solution=best_solution)
@info "results:" is_feasible(instance, sol) objective_value(instance, sol)
