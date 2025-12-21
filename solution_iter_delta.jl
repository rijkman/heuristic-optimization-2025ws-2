#!/usr/bin/env julia
include("instance_parsing.jl")
include("solution_iter.jl")
using Random, Combinatorics, DataStructures

#region ############## DELTA OPTIMIZATION ##############

PDPSwap = Vector{NTuple{4, Int64}}
PDPNeighborhood = BinaryMinHeap{Tuple{Float64,PDPSwap}}

function delta_objective_value_construct(
    instance::PDPInstance,
    solution::PDPSolutionVector,
    distances::Vector{Int64},
    route_k::Int64,
    loc_i::Int64,
)
    solution_k = solution[route_k]
    solution_init = length(solution_k) == 0
    # go from currect location (iter0: depot // iter1+: loc_i) to next and then back to depot
    distance_old_new = solution_init ? instance.distance_matrix[end, loc_i] : instance.distance_matrix[solution_k[end], loc_i]
    distance_home = instance.distance_matrix[loc_i, end]
    # calculate hypothetical costs
    distances[route_k] += (distance_old_new + distance_home)
    obj_val = sum(distances) + instance.ρ * (1 - jain_fairness(instance, distances))
    distances[route_k] -= (distance_old_new + distance_home) # avoid deepcopy by reversing
    return obj_val, distance_old_new
end

function delta_objective_value_improve(instance::PDPInstance, start_sol::PDPSolutionVector, swaps::Vector{Tuple{Int64,Int64,Int64,Int64}})
    distances::Vector{Int64} = []
    apply_swap!(start_sol, swaps)

    for route_k in start_sol
        distance = 0
        car_locations = length(route_k)
        if car_locations > 0
            for i in 1:car_locations-1
                distance += instance.distance_matrix[route_k[i], route_k[i+1]]
            end
            # add distances for travelling to and from depot
            distance += instance.distance_matrix[route_k[1], end]
            distance += instance.distance_matrix[route_k[end], end]
        end
        push!(distances, distance)
    end

    apply_swap!(start_sol, swaps)
    return sum(distances) + instance.ρ * (1 - jain_fairness(instance, distances))
end

function delta_is_feasible(instance::PDPInstance, satisfied_reqs::Vector{Int}, start_sol::PDPSolutionVector, swaps::Vector{Tuple{Int64,Int64,Int64,Int64}}, verbose=false)
    served_requests = 0
    affected_cars = falses(instance.n_vehicles)
    for (car_i, car_j, req_i, req_j) in swaps
        affected_cars[car_i] = true
        affected_cars[car_j] = true
        start_sol[car_i][req_i], start_sol[car_j][req_j] = start_sol[car_j][req_j], start_sol[car_i][req_i]
    end
    # requirement 1: vehicle capacity must never be exceeded at any point along route
    for route_k_idx in 1:instance.n_vehicles
        if !affected_cars[route_k_idx]
            served_requests += satisfied_reqs[route_k_idx]
            continue #skip this
        end
        route_k = start_sol[route_k_idx]
        visited_k = falses(instance.n_requests * 2)
        load_k = 0
        for loc_i in route_k
            pickup_i, dropoff_i, is_pickup, load_i = instance.requests[loc_i]
            if is_pickup
                load_k += load_i
                visited_k[loc_i] = true
                if load_k > instance.capacity
                    if verbose
                        @warn "Exceeded Capacity!"
                    end
                    apply_swap!(start_sol, swaps) #swap back
                    return false
                end
            else # is dropoff
                if visited_k[pickup_i]
                    load_k -= load_i
                    visited_k[pickup_i] = false
                    served_requests += 1 # visited both pickup and dropoff
                else
                    if verbose
                        @warn "Vehicle unnecessarily visiting drop-off $loc_i !"
                    end
                    # apply_swap!(start_sol, swaps) #swap back
                    # return false # theoretically incorrect but always leads to worse sols
                end
            end
        end
    end

    # requirement 3: at least γ requests must be served across all vehicles
    if served_requests < instance.γ
        if verbose
            @warn "Not enough requests served - $served_requests, out of $(instance.γ)!"
        end
        apply_swap!(start_sol, swaps) #swap back
        return false
    end
    # requirement 2: each served request must be handled entirely by single vehicle

    owner = fill(0, instance.n_requests * 2)
    for k in 1:instance.n_vehicles
        if affected_cars[k]
            for r in start_sol[k]
                if owner[r] != 0
                    apply_swap!(start_sol, swaps) #swap back
                    @warn "Overlapping routes"
                    return false
                else
                    owner[r] = 1
                end
            end
        end
    end

    apply_swap!(start_sol, swaps) #swap back
    return true
end

function apply_swap!(sol::PDPSolutionVector, swaps::Vector{Tuple{Int64,Int64,Int64,Int64}})
    for (car_i, car_j, req_i, req_j) in swaps
        sol[car_i][req_i], sol[car_j][req_j] = sol[car_j][req_j], sol[car_i][req_i]
    end
    return sol
end

function pc_served_requests(instance::PDPInstance, solution::PDPSolutionVector)::Vector{Int}
    served_reqs_total = Vector{Int}()

    # requirement 1: vehicle capacity must never be exceeded at any point along route
    for route_k in solution # for every car
        visited_k::Vector{Int64} = []
        served_requests = 0
        load_k = 0
        for loc_i in route_k
            pickup_i, dropoff_i, is_pickup, load_i = instance.requests[loc_i]
            if is_pickup
                load_k += load_i
                push!(visited_k, loc_i)
                if load_k > instance.capacity
                    error("Precomputed Served Requests Exceeded Load")
                end
            else # is dropoff
                if pickup_i in visited_k
                    load_k -= load_i
                    deleteat!(visited_k, findfirst(==(pickup_i), visited_k))
                    served_requests += 1 # visited both pickup and dropoff
                end
            end
        end
        push!(served_reqs_total, served_requests)
    end
    return served_reqs_total
end

#endregion

#region ############## NEIGHBORHOODS ##############

function delta_get_neighbor_solution(
    instance::PDPInstance,
    solution::PDPSolutionVector,
    score::Float64,
    neighborhood_func::Function,
    step_func::Function,
)
    solution_neighbors = neighborhood_func(instance, solution)
    best_swaps, best_score = step_func(solution_neighbors, solution, score)
    best_solution = apply_swap!(deepcopy(solution), best_swaps)
    return best_solution, best_score
end

function delta_step_best(
    solution_neighbors::PDPNeighborhood,
    solution::PDPSolutionVector,
    score::Float64,
)
    score, solution = length(solution_neighbors) > 0 ? first(solution_neighbors) : (score, PDPSwap())
    return (solution, score)
end

function delta_step_random(
    solution_neighbors::PDPNeighborhood,
    solution::PDPSolutionVector,
    score::Float64,
    alpha::Float64=0.3
)
    score, solution = length(solution_neighbors) > 0 ? rand([pop!(solution_neighbors) for _ in 1:alpha*length(solution_neighbors)]) : (score, PDPSwap())
    return solution, score
end

function delta_neighbor_in_switch_location(instance::PDPInstance, solution::PDPSolutionVector)
    neigborhood = PDPNeighborhood()
    #prepopulate 
    served_requests = pc_served_requests(instance, solution)

    swaps = Vector{Tuple{Int,Int,Int,Int}}()
    for k in 1:instance.n_vehicles
        route_k = solution[k]
        route_kn = length(route_k)
        # try switching every possible location pair
        for i in 1:route_kn-1
            for j in i+1:route_kn
                swaps = [(k, k, i, j)]
                if delta_is_feasible(instance, served_requests, solution, swaps)
                    score = delta_objective_value_improve(instance, solution, swaps)
                    push!(neigborhood, (score, [(k, k, i, j)]))
                end
            end
        end
    end
    return neigborhood
end

function delta_neighbor_in_subsequence(instance::PDPInstance, solution::PDPSolutionVector)
    neighborhood = PDPNeighborhood()
    served_requests = pc_served_requests(instance, solution)
    max_subsequence_lengths = 5:8
    swaps = Vector{Tuple{Int,Int,Int,Int}}()

    for k in 1:instance.n_vehicles
        route_k = solution[k]
        n = length(route_k)
        for l in max_subsequence_lengths
            #start of subsequence_1
            for i in 1:(n-2l+1)
                #start of subsequence_2
                for j in i+l:(n-l+1)
                    #swap subsequences
                    swaps = [
                        (k, k, i + t, j + t)
                        for t in 0:(l-1)
                    ]
                    if delta_is_feasible(instance, served_requests, solution, swaps)
                        score = delta_objective_value_improve(instance, solution, swaps)
                        push!(neighborhood, (score, swaps))
                    end
                end
            end
        end
    end

    return neighborhood

end

function delta_neighbor_between_switch_request(instance::PDPInstance, solution::Vector{Vector{Int64}})
    """
    Returns all feasible solutons, where one pair
    in the following format:
    (score, car_i, pickup_i, dropoff_i, car_j, pickup_j, dropoff_j)
    """
    neighborhood = PDPNeighborhood()

    served_reqs = pc_served_requests(instance, solution)
    # get (pickup_index_i, dropoff_index_i)
    # assumes all pairs are completed
    all_fulfilled_requests::Vector{Vector{Tuple{Int64,Int64}}} = []
    tracked_pickups = Dict{Int64,Int64}()
    for k in 1:instance.n_vehicles
        fulfilled_requests = []
        for i in 1:length(solution[k])
            _, _, pickup_flag, _ = instance.requests[solution[k][i]]
            if pickup_flag
                tracked_pickups[solution[k][i]+instance.n_requests] = i
            else
                push!(
                    fulfilled_requests,
                    (tracked_pickups[solution[k][i]], i)
                )
            end
        end
        push!(all_fulfilled_requests, fulfilled_requests)
    end

    swaps = Vector{Tuple{Int,Int,Int,Int}}()
    # go over all possible combinations
    for car_i in 1:instance.n_vehicles
        for car_j in car_i:instance.n_vehicles
            for (pickup_i, dropoff_i) in all_fulfilled_requests[car_i]
                for (pickup_j, dropoff_j) in all_fulfilled_requests[car_j]
                    swaps = [
                        (car_i, car_j, pickup_i, pickup_j),
                        (car_i, car_j, dropoff_i, dropoff_j)
                    ]
                    if delta_is_feasible(instance, served_reqs, solution, swaps)
                        score = delta_objective_value_improve(instance, solution, swaps)
                        push!(neighborhood, (
                            score, swaps
                        ))
                    end
                end
            end
        end
    end
    return neighborhood
end

#endregion