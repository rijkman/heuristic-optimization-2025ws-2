#!/usr/bin/env julia
include("solution_iter_delta.jl")
include("greedy.jl")

function greedy_insertion(
    instance::PDPInstance,
    partial_solution::PDPSolutionVector,
    verbose::Bool,
)
    # initialize variables
    dm = instance.distance_matrix
    master_solution = deepcopy(partial_solution)
    master_distances = [0 for _ in 1:instance.n_vehicles]
    
    master_loads = [Int[] for _ in 1:instance.n_vehicles]
    for vehicle_k in 1:instance.n_vehicles
        current_load = 0
        for loc in partial_solution[vehicle_k]
            _, _, is_pickup, load = instance.requests[loc]
            current_load = max(0, is_pickup ? current_load + load : current_load - load)
            push!(master_loads[vehicle_k], current_load)
        end
    end

    open_requests = union(getindex.(instance.requests, 1), getindex.(instance.requests, 2))
    open_requests = setdiff!(open_requests, vcat(partial_solution...))
    served_requests = sum(pc_served_requests(instance, partial_solution))

    
    while served_requests < instance.γ
        best_candidate = (0,0,0,0)
        best_candidate_score = Inf
        # determine best candidate
        for loc_i in open_requests #for all open requests
            pickup_idx, dropoff_idx, is_pickup, load = instance.requests[loc_i]
            
            for vehicle_k in 1:instance.n_vehicles # hypothetically assign to any vehicle
                k_n = length(master_solution[vehicle_k])
                if is_pickup
                    # edge case: empty route
                    if k_n == 0
                        #calculate hypothetical objective value if inserting at this place
                        hyp_obj_value, distance_diff = insert_obj_value(
                            instance,
                            master_solution,
                            master_distances,
                            vehicle_k,
                            loc_i,
                            1
                        )
                        
                        if hyp_obj_value < best_candidate_score
                            best_candidate = (vehicle_k, loc_i, 1, distance_diff)
                            best_candidate_score = hyp_obj_value
                        end
                        continue
                    else
                
                        for insert_idx in 1:k_n+1 # at any given position 
                            pre_idx = insert_idx == k_n+1 ? k_n : insert_idx                    
                            if master_solution[vehicle_k][pre_idx] == dropoff_idx
                                # cannot place pickup after dropoff
                                break
                            end

                            if maximum(master_loads[vehicle_k][max(1, pre_idx-1):end]) + load > instance.capacity
                                continue
                            end

                            #calculate hypothetical objective value if inserting at this place
                            hyp_obj_value, distance_diff = insert_obj_value(
                                instance,
                                master_solution,
                                master_distances,
                                vehicle_k,
                                loc_i,
                                insert_idx
                            )
                            
                            if hyp_obj_value < best_candidate_score
                                best_candidate = (vehicle_k, loc_i, insert_idx, distance_diff)
                                best_candidate_score = hyp_obj_value
                            end
                        end
                    end
                else
                    if k_n == 0 # edge case, empty route
                        continue #skip this one
                    end

                    has_picked_up = false
                    for insert_idx in 1:k_n+1 # at any given position 
                        pre_idx = insert_idx == k_n+1 ? k_n : insert_idx  
                        if master_solution[vehicle_k][pre_idx] == pickup_idx
                            has_picked_up = true
                            continue # can insert on the next iteration
                        end
                        
                        # check whether this load was already picked up
                        if !has_picked_up
                            continue
                        end

                        #calculate hypothetical value of inserting here
                        hyp_obj_value, distance_diff = insert_obj_value(
                            instance,
                            master_solution,
                            master_distances,
                            vehicle_k,
                            loc_i,
                            insert_idx
                        )

                        if hyp_obj_value < best_candidate_score
                            best_candidate = (vehicle_k, loc_i, insert_idx, distance_diff)
                            best_candidate_score = hyp_obj_value
                        end
                    end
                end
            end
        end

        if best_candidate_score == Inf
            error("No feasible solution found")
        end

        
        # apply best candidate
        best_vehicle_k, best_loc_i, best_insert_idx, distance_diff = best_candidate
        pickup_idx, dropoff_idx, is_pickup, load = instance.requests[best_loc_i] 
        insert!(master_solution[best_vehicle_k], best_insert_idx, best_loc_i)

        master_distances[best_vehicle_k] += distance_diff
        prev_load = best_insert_idx == 1 ? 0 : master_loads[best_vehicle_k][best_insert_idx-1]
        if is_pickup
            insert!(master_loads[best_vehicle_k], best_insert_idx, prev_load+load)
        else
            insert!(master_loads[best_vehicle_k], best_insert_idx, max(0,prev_load-load))
        end
        for j in best_insert_idx+1:length(master_loads[best_vehicle_k])
            _, _, j_is_pickup, load = instance.requests[master_solution[best_vehicle_k][j]]
            if j_is_pickup
                master_loads[best_vehicle_k][j] = master_loads[best_vehicle_k][j-1] + load
            else
                master_loads[best_vehicle_k][j] = max(0, master_loads[best_vehicle_k][j-1] - load)
            end
        end

        if !is_pickup
            served_requests += 1 #TODO: probably need to improve this logic
        end
        deleteat!(open_requests, findfirst(==(best_loc_i), open_requests))
    end
    return master_solution
end

function calculate_loads(
    instance::PDPInstance,
    solution::PDPSolutionVector
)
    loads = [Int[] for _ in 1:instance.n_vehicles]
    for k in 1:instance.n_vehicles
        current_load = 0
        for req in solution[k]
            _,_,is_pickup,load = instance.requests[req]
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

function insert_obj_value(
    instance::PDPInstance, 
    solution::PDPSolutionVector, 
    distances::Vector{Int64},
    route_k::Int64,
    loc_i::Int64,
    insert_idx::Int64
)   
    solution_k = solution[route_k]
    n_k = length(solution_k)
    dm = instance.distance_matrix
    distance_diff = 0

    depot_index, _  = size(dm)
    pre = insert_idx == 1 ? depot_index : solution_k[insert_idx-1]
    suc = insert_idx > n_k ? depot_index : solution_k[insert_idx]

    distance_diff = dm[pre, loc_i] + dm[loc_i, suc] - dm[pre, suc]

    distances[route_k] += distance_diff
    obj_val = sum(distances) + instance.ρ * (1 - jain_fairness(instance, distances))
    distances[route_k] += -distance_diff # reverse changes (to avoid deepcopy)
    return obj_val, distance_diff
end

instance = read_PDPInstance("instances/2000/competition/", "instance61_nreq2000_nveh40_gamma1829.txt")
# print(instance.requests)
sol = greedy_insertion(instance, [Vector{Int64}() for _ in 1:instance.n_vehicles], false)
@info "max capacity" instance.capacity
@info "eval" is_feasible(instance, sol, true) objective_value(instance, sol)
@info "served reqs" pc_served_requests(instance, sol) sum(pc_served_requests(instance, sol))
# visualize(instance, sol, 0.0)

# greedy_heuristic_one_extend(instance)