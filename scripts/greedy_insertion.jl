#!/usr/bin/env julia
include("greedy.jl")

function greedy_heuristic_insertion(
    instance::PDPInstance;
    fairness::Function=jain_fairness,
    verbose::Bool=true,
)
    solution = [Int64[] for _ in 1:instance.n_vehicles]
    return greedy_insertion(
        fairness,
        instance,
        solution,
        verbose
    )
end

function greedy_insertion(
    fairness::Function,
    instance::PDPInstance,
    partial_solution::PDPSolutionVector,
    verbose::Bool=true,
)
    # initialize variables from partial
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

    # run actual heuristic
    iter_n = 0
    iter_score = Vector{Float64}()
    while served_requests < instance.γ
        best_candidate = (0, 0, 0, 0)
        best_candidate_score = Inf
        # determine best candidate
        for loc_i in open_requests # for all open requests
            pickup_idx, dropoff_idx, is_pickup, load = instance.requests[loc_i]
            for vehicle_k in 1:instance.n_vehicles # hypothetically assign to any vehicle
                k_n = length(master_solution[vehicle_k])
                if is_pickup
                    # edge case: empty route
                    if k_n == 0
                        # calculate hypothetical objective value if inserting at this place
                        hyp_obj_value, distance_diff = delta_objective_value_insert(
                            fairness,
                            instance,
                            master_solution,
                            master_distances,
                            vehicle_k,
                            loc_i,
                            1,
                        )
                        if hyp_obj_value < best_candidate_score
                            best_candidate = (vehicle_k, loc_i, 1, distance_diff)
                            best_candidate_score = hyp_obj_value
                        end
                    else
                        for insert_idx in 1:k_n+1 # at any given position 
                            pre_idx = insert_idx == k_n + 1 ? k_n : insert_idx
                            if master_solution[vehicle_k][pre_idx] == dropoff_idx
                                # cannot place pickup after dropoff
                                break
                            end

                            if maximum(master_loads[vehicle_k][max(1, pre_idx - 1):end]) + load > instance.capacity
                                continue
                            end

                            # calculate hypothetical objective value if inserting at this place
                            hyp_obj_value, distance_diff = delta_objective_value_insert(
                                fairness,
                                instance,
                                master_solution,
                                master_distances,
                                vehicle_k,
                                loc_i,
                                insert_idx,
                            )

                            if hyp_obj_value < best_candidate_score
                                best_candidate = (vehicle_k, loc_i, insert_idx, distance_diff)
                                best_candidate_score = hyp_obj_value
                            end
                        end
                    end
                else
                    if k_n == 0 # edge case, empty route
                        continue # skip this one
                    end

                    has_picked_up = false
                    for insert_idx in 1:k_n+1 # at any given position 
                        pre_idx = insert_idx == k_n + 1 ? k_n : insert_idx
                        if master_solution[vehicle_k][pre_idx] == pickup_idx && !has_picked_up # last element
                            has_picked_up = true
                            continue # can insert on the next iteration
                        end

                        # check whether this load was already picked up
                        if !has_picked_up
                            continue
                        end

                        # calculate hypothetical value of inserting here
                        hyp_obj_value, distance_diff = delta_objective_value_insert(
                            fairness,
                            instance,
                            master_solution,
                            master_distances,
                            vehicle_k,
                            loc_i,
                            insert_idx,
                        )

                        if hyp_obj_value < best_candidate_score
                            best_candidate = (vehicle_k, loc_i, insert_idx, distance_diff)
                            best_candidate_score = hyp_obj_value
                        end
                    end
                end
            end
        end

        # apply best candidate
        best_vehicle_k, best_loc_i, best_insert_idx, distance_diff = best_candidate
        pickup_idx, dropoff_idx, is_pickup, load = instance.requests[best_loc_i]
        insert!(master_solution[best_vehicle_k], best_insert_idx, best_loc_i)

        master_distances[best_vehicle_k] += distance_diff
        prev_load = best_insert_idx == 1 ? 0 : master_loads[best_vehicle_k][best_insert_idx-1]
        if is_pickup
            insert!(master_loads[best_vehicle_k], best_insert_idx, prev_load + load)
        else
            insert!(master_loads[best_vehicle_k], best_insert_idx, max(0, prev_load - load))
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
            served_requests += 1
        end
        deleteat!(open_requests, findfirst(==(best_loc_i), open_requests))
        push!(iter_score, best_candidate_score)
        log_iteration(iter_n, best_candidate_score, verbose)
        iter_n += 1
    end
    master_score = objective_value(fairness, instance, master_solution)
    log_result(instance, master_solution, master_score, verbose)
    return (iter_score, iter_n, master_solution, master_score)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    test_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_insertion,
    )
end