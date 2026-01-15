#!/usr/bin/env julia
include("greedy_random.jl")

function ant_colony_system(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    fairness::Function=jain_fairness,
    # runtime parameters
    n_iterations::Int64=200,
    m_colony::Int64=50,
    # pheromone evaporation coefficients
    local_phero_decay::Float64=0.05,
    phero_decay::Float64=0.1,
    # state transition coefficients
    beta::Float64=3.0,
    alpha::Float64=1.0,
    factor_greed::Float64=0.9,
    # defaults
    verbose::Bool=true,
)
    n_locations = length(instance.locations)
    # desirability of state transition // a priori
    attract_level_demo = 1 ./ instance.distance_matrix # must calculate per iteration on-the-fly 
    # amount of pheromones // a posteriori
    phero_init = 1 / (n_locations * score)
    phero_level = ones(n_locations, n_locations) * phero_init

    iter_n = 0 # any loop
    iter_score = Vector{Float64}()
    best_score = Inf
    best_solution = solution
    curr_iter = 0 # outer loop
    while curr_iter < n_iterations # iterate over n tries across m ants
        for k in 1:m_colony
            # keep track of solution [analogous to greedy scheme]
            ant_score = Inf
            ant_solution = [Int64[] for _ in 1:instance.n_vehicles]
            ant_distances = [0 for _ in 1:instance.n_vehicles]
            ant_loads = [0 for _ in 1:instance.n_vehicles]
            open_pickups = Set(collect(1:instance.n_requests))
            open_dropoffs = [Int64[] for _ in 1:instance.n_vehicles]
            served_requests = 0

            # start with explorative random move
            start_route_k = rand(1:instance.n_vehicles)
            start_loc_i = rand(open_pickups)
            _, start_distance = delta_objective_value_construct(fairness, instance, ant_solution, ant_distances, start_route_k, start_loc_i)
            solution_step_update!(
                instance,
                ant_solution,
                ant_distances,
                ant_loads,
                start_route_k,
                start_loc_i,
                start_distance,
                open_pickups,
                open_dropoffs,
            )

            # construct vehicle tour [analogous to greedy scheme]
            while served_requests < instance.γ
                # define greedy vs random decision across vehicles
                select_greed = rand() <= factor_greed
                select_list = Vector{Tuple{Int64,Int64,Float64,Int64}}() # route_k, loc_i, score, distance

                # calculate attractiveness level on-the-fly for each iteration
                attract_level = zeros(n_locations, n_locations) # scores 
                for route_k in 1:instance.n_vehicles
                    route_curr_loc = length(route_k) == 0 ? n_locations : route_k[end]
                    route_open_locs = vcat(open_pickups..., open_dropoffs[route_k]...)
                    route_cand_locs = Vector{Int64}() # loc_i
                    route_cand_step = Vector{Tuple{Int64,Int64,Float64,Int64}}() # route_k, loc_i, score, distance
                    for loc_i in route_open_locs
                        _, _, is_pickup, load_i = instance.requests[loc_i]
                        can_visit = (ant_loads[route_k] + load_i) <= instance.capacity || !is_pickup
                        if can_visit
                            curr_score, curr_distance = delta_objective_value_construct(fairness, instance, ant_solution, ant_distances, route_k, loc_i)
                            attract_level[route_curr_loc, loc_i] = 1 / curr_score
                            push!(route_cand_locs, loc_i)
                            push!(route_cand_step, (route_k, loc_i, curr_score, curr_distance))
                        end
                    end

                    # calculate state transition probability per vehicle
                    if length(route_cand_locs) > 0
                        route_trans_probs = state_trans_prob(route_curr_loc, route_cand_locs, phero_level, attract_level, alpha, beta)
                        route_next_loc_idx = select_greed ? trans_greedy(route_trans_probs) : trans_prob(route_trans_probs)
                        route_next = route_cand_step[route_next_loc_idx]
                        push!(select_list, route_next)
                    end
                end

                # select random step across greedy/random selections of vehicles (vs greedy)
                ant_route, ant_loc, ant_score, ant_distance = rand(select_list) # argmax(x -> x[3], select_list)
                served_requests += solution_step_update!(
                    instance,
                    ant_solution,
                    ant_distances,
                    ant_loads,
                    ant_route,
                    ant_loc,
                    ant_distance,
                    open_pickups,
                    open_dropoffs,
                )
            end
            update_pheromones_local!(ant_solution, phero_level, local_phero_decay, phero_init)
            if ant_score < best_score
                best_score = ant_score
                best_solution = ant_solution
            end
            push!(iter_score, best_score)
            log_iteration(iter_n, best_score, verbose)
            iter_n += 1
        end
        update_pheromones_global!(best_solution, best_score, phero_level, phero_decay)
        curr_iter += 1
    end
    best_score = objective_value(fairness, instance, best_solution)
    log_result(instance, best_solution, best_score, verbose)
    return (iter_score, iter_n, best_solution, best_score)
end

function state_trans_prob(
    route_curr_loc::Int64,
    route_open_locs::Vector{Int64},
    phero_level::Matrix{Float64},
    attract_level::Matrix{Float64},
    alpha::Float64,
    beta::Float64,
)
    i = route_curr_loc
    total_level_norm = 0
    route_trans_probs = zeros(length(route_open_locs))
    for l in route_open_locs
        total_level_norm += (phero_level[i, l]^alpha) * (attract_level[i, l]^beta)
    end
    for (idx, j) in enumerate(route_open_locs)
        route_trans_probs[idx] = (phero_level[i, j]^alpha) * (attract_level[i, j]^beta) / total_level_norm
    end
    return route_trans_probs
end

function trans_greedy(route_trans_probs::Vector{Float64})
    return argmax(route_trans_probs)
end

function trans_prob(route_trans_probs::Vector{Float64})
    return sample(CartesianIndices(route_trans_probs), weights(route_trans_probs))[1]
end

function update_pheromones_local!(
    ant_solution::PDPSolutionVector,
    phero_level::Matrix{Float64},
    local_phero_decay::Float64,
    phero_init::Float64,
)
    for route_k in ant_solution
        for (i, j) in partition(route_k, 2, 1)
            phero_update = (1 - local_phero_decay) * phero_level[i, j] + local_phero_decay * phero_init
            phero_level[i, j] = phero_update
        end
    end
end

function update_pheromones_global!(
    best_solution::PDPSolutionVector,
    best_score::Float64,
    phero_level::Matrix{Float64},
    phero_decay::Float64,
)
    phero_level = phero_level .* (1 - phero_decay)
    for route_k in best_solution
        for (i, j) in partition(route_k, 2, 1)
            phero_delta = 1 / best_score
            phero_level[i, j] += phero_decay * phero_delta
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    _, _, solution, score = solve_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend,
        Dict(:verbose => false)...
    )
    test_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=ant_colony_system,
        Dict(:solution => solution, :score => score)...,
    )
end