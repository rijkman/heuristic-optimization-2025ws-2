#!/usr/bin/env julia
include("logging.jl")

#region ############## FAIRNESS ##############

function jain_fairness(instance::PDPInstance, distances::Vector{Int64})
    nom = sum(distances)^2
    denom = sum(map(rk -> rk^2, distances)) * instance.n_vehicles
    return denom == 0 ? 0 : nom / denom
end

function max_min_fairness(instance::PDPInstance, distances::Vector{Int64})
    nom = minimum(distances)
    denom = maximum(distances)
    return denom == 0 ? 0 : nom / denom
end

function gini_fairness(instance::PDPInstance, distances::Vector{Int64})
    nom = sum([abs(rk - rkk) for rk in distances, rkk in distances])
    denom = sum(distances) * 2instance.n_vehicles
    return denom == 0 ? 0 : nom / denom
end

function objective_formula(
    fairness::Function,
    instance::PDPInstance,
    distances::Vector{Int64}
)
    return sum(distances) + instance.ρ * (1 - fairness(instance, distances))
end

#endregion

#region ############## OPTIMIZATION ##############

function objective_value(
    fairness::Function,
    instance::PDPInstance,
    solution::PDPSolutionVector
)
    distances = calculate_distances(instance, solution)
    return objective_formula(fairness, instance, distances)
end

function calculate_distances(
    instance::PDPInstance,
    solution::PDPSolutionVector
)
    distances::Vector{Int64} = []
    for route_k in solution # for every car
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
    return distances
end

function calculate_loads(
    instance::PDPInstance,
    solution::PDPSolutionVector
)
    loads = [Int[] for _ in 1:instance.n_vehicles]
    for k in 1:instance.n_vehicles
        current_load = 0
        for req in solution[k]
            _, _, is_pickup, load = instance.requests[req]
            if is_pickup
                current_load += load
            else
                current_load = max(0, current_load - load)
            end
            push!(loads[k], current_load)
        end
    end
    return loads
end

function is_feasible(
    instance::PDPInstance,
    solution::PDPSolutionVector,
    verbose::Bool=true
)
    served_requests = 0

    # requirement 1: vehicle capacity must never be exceeded at any point along route
    for route_k in solution # for every car
        visited_k::Vector{Int64} = []
        load_k = 0
        for loc_i in route_k
            pickup_i, dropoff_i, is_pickup, load_i = instance.requests[loc_i]
            if is_pickup
                load_k += load_i
                push!(visited_k, loc_i)
                if load_k > instance.capacity
                    if verbose
                        @warn "Exceeded Capacity!"
                    end
                    return false
                end
            else # is dropoff
                if pickup_i in visited_k
                    load_k -= load_i
                    deleteat!(visited_k, findfirst(==(pickup_i), visited_k))
                    served_requests += 1 # visited both pickup and dropoff
                else
                    if verbose
                        @warn "Vehicle unnecessarily visiting drop-off $loc_i !"
                    end
                end
            end
        end
    end

    # requirement 3: at least γ requests must be served across all vehicles
    if served_requests < instance.γ
        if verbose
            @warn "Not enough requests served - $served_requests, out of $(instance.γ)!"
        end
        return false
    end

    # requirement 2: each served request must be handled entirely by single vehicle
    for i in 1:instance.n_vehicles
        for j in i+1:instance.n_vehicles
            if length(intersect(solution[i], solution[j])) > 0 # check pairwise intersection
                if verbose
                    @warn "Overlapping routes!"
                end
                return false
            end
        end
    end

    return true
end

#endregion

#region ############## NEIGHBORHOODS ##############

# neighborhood and step function structs
function get_neighbor_solution(
    fairness::Function,
    instance::PDPInstance,
    solution::PDPSolutionVector,
    score::Float64,
    neighborhood_func::Function,
    step_func::Function,
)
    solution_neighbors = neighborhood_func(instance, solution)
    best_solution, best_score = step_func(fairness, instance, solution, score, solution_neighbors)
    return best_solution, best_score
end

function step_first(
    fairness::Function,
    instance::PDPInstance,
    solution::PDPSolutionVector,
    score::Float64,
    solution_neighbors::Set{PDPSolutionVector},
)
    best_solution = solution
    best_score = score
    for curr_neighbor in solution_neighbors
        curr_score = objective_value(fairness, instance, curr_neighbor)
        if curr_score < best_score
            best_score = curr_score
            best_solution = curr_neighbor
            break # break after first improved neighbor
        end
    end
    return best_solution, best_score
end

function step_best(
    fairness::Function,
    instance::PDPInstance,
    solution::PDPSolutionVector,
    score::Float64,
    solution_neighbors::Set{PDPSolutionVector},
)
    best_solution = solution
    best_score = score
    for curr_neighbor in solution_neighbors
        curr_score = objective_value(fairness, instance, curr_neighbor)
        if curr_score < best_score
            best_score = curr_score
            best_solution = curr_neighbor
            # iterate over all neighbors
        end
    end
    return best_solution, best_score
end

function step_random(
    fairness::Function,
    instance::PDPInstance,
    solution::PDPSolutionVector,
    score::Float64,
    solution_neighbors::Set{PDPSolutionVector},
)
    best_solution = rand(solution_neighbors) # choose random neighbor
    best_score = objective_value(fairness, instance, best_solution)
    return best_solution, best_score
end

# in car: switch any two locations
function neighbor_in_switch_location(instance::PDPInstance, solution::PDPSolutionVector)
    solution_neighbors = Set{PDPSolutionVector}()
    # for every car
    for k in 1:instance.n_vehicles
        route_k = solution[k]
        route_kn = length(route_k)
        # try switching every possible location pair
        for i in 1:route_kn-1
            for j in i+1:route_kn
                solution_switch = deepcopy(solution)
                solution_switch[k][i], solution_switch[k][j] = solution_switch[k][j], solution_switch[k][i]
                if is_feasible(instance, solution_switch, false)
                    push!(solution_neighbors, solution_switch)
                end
            end
        end
    end
    return solution_neighbors
end

# between cars: move request to other car
function neighbor_between_move_request(instance::PDPInstance, solution::PDPSolutionVector)
    solution_neighbors = Set{PDPSolutionVector}()
    # for every different pair of cars
    for k in 1:instance.n_vehicles
        route_k = solution[k]
        for l in 1:instance.n_vehicles
            route_l = solution[l]
            route_ln = length(route_l)
            if k != l
                # for every request in route of car k
                for loc_i in route_k
                    pickup_i, dropoff_i, is_pickup, _ = instance.requests[loc_i]
                    if is_pickup
                        # try moving request to any possible spot in route of car l
                        for j_pickup in 1:route_ln
                            for j_dropoff in j_pickup+1:route_ln+1
                                solution_move = deepcopy(solution)
                                deleteat!(solution_move[k], findfirst(==(pickup_i), solution_move[k]))
                                deleteat!(solution_move[k], findfirst(==(dropoff_i), solution_move[k]))
                                insert!(solution_move[l], j_pickup, pickup_i)
                                insert!(solution_move[l], j_dropoff, dropoff_i)
                                if is_feasible(instance, solution_move, false)
                                    push!(solution_neighbors, solution_move)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return solution_neighbors
end

# in car: try multiple random location permutations
function neighbor_in_permutate_location_single(instance::PDPInstance, solution::PDPSolutionVector)
    solution_neighbors = Set{PDPSolutionVector}()
    switch_choices = 2:5
    while true
        # for one random car
        k = rand(1:instance.n_vehicles)
        route_k = solution[k]
        route_kn = length(route_k)
        # for random number of swaps
        switch_count = min(rand(switch_choices), route_kn)
        # select switch_count locations
        choice_indices_k = collect(combinations(1:route_kn, switch_count))
        choice_i = rand(choice_indices_k)
        choice_values_i = route_k[choice_i]
        # and permutate selected locations
        perm_values_i = permutations(choice_values_i)
        perm_i = rand(collect(perm_values_i))
        solution_perm = deepcopy(solution)
        for (i, idx) in enumerate(choice_i)
            solution_perm[k][idx] = perm_i[i]
        end
        # check new solution
        if solution_perm[k] != solution[k]
            if is_feasible(instance, solution_perm, false)
                push!(solution_neighbors, solution_perm)
                break
            end
        end
    end
    return solution_neighbors
end

# in car: try any possible location permutation - full version of above; too inefficient; delete?
function neighbor_in_permutate_location(instance::PDPInstance, solution::PDPSolutionVector)
    solution_neighbors = Set{PDPSolutionVector}()
    switch_count = 3
    # for every car
    for k in 1:instance.n_vehicles
        route_k = solution[k]
        route_kn = length(route_k)
        # select switch_count locations
        choice_indices_k = collect(combinations(1:route_kn, switch_count))
        for choice_i in choice_indices_k
            choice_values_i = route_k[choice_i]
            # and permutate selected locations
            perm_values_i = permutations(choice_values_i)
            for perm_i in perm_values_i
                solution_perm = deepcopy(solution)
                for (i, idx) in enumerate(choice_i)
                    solution_perm[k][idx] = perm_i[i]
                    if is_feasible(instance, solution_perm, false)
                        push!(solution_neighbors, solution_perm)
                    end
                end
            end
        end
    end
    return solution_neighbors
end

#endregion