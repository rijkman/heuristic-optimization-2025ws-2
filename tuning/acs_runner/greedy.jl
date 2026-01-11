#!/usr/bin/env julia
include("solution_iter_delta.jl")

function greedy_heuristic_one_extend(
    instance::PDPInstance;
    lookahead_gamma::Union{Int64,Nothing}=nothing,
    verbose::Bool=true
)
    solution = [Int64[] for _ in 1:instance.n_vehicles]
    distances = [0 for _ in 1:instance.n_vehicles]
    loads = [0 for _ in 1:instance.n_vehicles]
    open_pickups = Set(collect(1:instance.n_requests))
    open_dropoffs = [Int64[] for _ in 1:instance.n_vehicles]
    served_requests = 0
    return greedy_heuristic_from_solution(
        instance,
        solution,
        distances,
        loads,
        open_pickups,
        open_dropoffs,
        served_requests,
        lookahead_gamma,
        verbose,
    )
end

function greedy_heuristic_from_solution(
    instance::PDPInstance,
    partial_solution::PDPSolutionVector
)
    distances = Vector{Int64}()
    loads = Vector{Int64}()
    for vehicle_k in 1:instance.n_vehicles
        t_dist = 0
        t_load = 0
        if length(partial_solution[vehicle_k]) == 0
            push!(distances, 0)
            push!(loads, 0)
            continue
        end
        for (i, req) in enumerate(partial_solution[vehicle_k])
            _, _, is_pickup, load = instance.requests[req]
            t_load = max(0, is_pickup ? t_load + load : t_load - load)
            prev = i == 1 ? length(instance.distance_matrix[1]) : partial_solution[vehicle_k][i-1]
            t_dist += instance.distance_matrix[req, prev]
        end
        t_dist += instance.distance_matrix[partial_solution[vehicle_k][end], end]
        push!(distances, t_dist)
        push!(loads, t_load)
        
    end

    open_pickups = Set{Int64}()
    open_dropoffs = [Vector{Int64}() for _ in 1:instance.n_vehicles]
    scheduled_reqs = Dict()
    for k in 1:instance.n_vehicles
        for req in partial_solution[k]
            scheduled_reqs[req] = k
        end
    end
    
    for req in instance.requests
        pickup_idx, dropoff_idx, is_pickup, _ = req
        if is_pickup
            if !haskey(scheduled_reqs, pickup_idx)
                push!(open_pickups, pickup_idx)
            elseif !haskey(scheduled_reqs, dropoff_idx)
                k = scheduled_reqs[pickup_idx]
                push!(open_dropoffs[k], dropoff_idx)
            end
        end
    end
    served_requests = sum(pc_served_requests(instance, partial_solution))

    return greedy_heuristic_from_solution(
        instance,
        partial_solution,
        distances,
        loads,
        open_pickups,
        open_dropoffs,
        served_requests,
        nothing,
        false
    )
end

function greedy_heuristic_from_solution(
    instance::PDPInstance,
    best_solution::PDPSolutionVector,
    best_distances::Vector{Int64},
    best_loads::Vector{Int64},
    open_pickups::Set{Int64},
    open_dropoffs::Vector{Vector{Int64}},
    served_requests::Int64,
    lookahead_gamma::Union{Int64,Nothing},
    verbose::Bool,
)
    iter_n = 0
    iter_score = Vector{Float64}()
    while served_requests < instance.γ
        # log_unsatisfied(served_requests, instance.γ, verbose)
        any_open_locs = vcat(open_pickups..., open_dropoffs...)
        if length(any_open_locs) == 0
            log_unfeasable(verbose)
            return best_solution
        end
        # loop over all possible solutions for [one-step location extensions]
        best_score = Inf64
        best_loc = -1
        best_car = -1
        best_distance = -1
        for route_k in 1:instance.n_vehicles
            open_locs = vcat(open_pickups..., open_dropoffs[route_k]...)
            for loc_i in open_locs
                # hypothetically assign ith location to kth car
                _, _, is_pickup, load_i = instance.requests[loc_i]
                can_visit = (best_loads[route_k] + load_i) <= instance.capacity || !is_pickup
                if can_visit
                    # check resulting objective score
                    curr_score, curr_distance = delta_objective_value_construct(
                        instance,
                        best_solution,
                        best_distances,
                        route_k,
                        loc_i,
                    )
                    if curr_score < best_score
                        best_score = curr_score
                        best_distance = curr_distance
                        best_car = route_k
                        best_loc = loc_i
                    end
                end
            end
        end
        # remove pickup or dropoff from remaining requests locations
        served_requests += solution_step_update!(
            instance,
            best_solution,
            best_distances,
            best_loads,
            best_car,
            best_loc,
            best_distance,
            open_pickups,
            open_dropoffs,
        )
        push!(iter_score, best_score)
        log_iteration(iter_n, best_score, verbose)
        iter_n += 1
        # quit after gamma steps if called by pilot
        if lookahead_gamma != nothing && iter_n == (lookahead_gamma - 1)
            break
        end
    end
    best_solution = solution_clean!(instance, best_solution)
    best_score = objective_value(instance, best_solution)
    log_result(instance, best_solution, best_score, verbose)
    return (iter_score, iter_n, best_solution, best_score)
end

function solution_step_update!(
    instance::PDPInstance,
    solution::PDPSolutionVector,
    distances::Vector{Int64},
    loads::Vector{Int64},
    route_k::Int64,
    loc_i::Int64,
    distance_ik::Int64,
    open_pickups::Set{Int64},
    open_dropoffs::Vector{Vector{Int64}},
)
    _, dropoff_i, is_pickup, load_i = instance.requests[loc_i]
    push!(solution[route_k], loc_i)
    distances[route_k] += distance_ik
    if is_pickup
        loads[route_k] += load_i
        pop!(open_pickups, loc_i)
        push!(open_dropoffs[route_k], dropoff_i)
        new_request = 0
    else # is dropoff
        loads[route_k] -= load_i
        new_request = 1 # visited both pickup and dropoff
        deleteat!(open_dropoffs[route_k], findfirst(==(dropoff_i), open_dropoffs[route_k]))
    end
    return new_request
end

function solution_clean!(
    instance::PDPInstance,
    best_solution::PDPSolutionVector,
)
    # remove any unnecessarily visited pickups without dropoffs; lone dropoffs are not possible because open_dropoffs
    for route_k in best_solution
        for loc_i in route_k
            pickup_i, dropoff_i, _, _ = instance.requests[loc_i]
            pickup_idx = findfirst(==(pickup_i), route_k)
            dropoff_idx = findfirst(==(dropoff_i), route_k)
            if dropoff_idx == nothing
                deleteat!(route_k, pickup_idx)
            end
        end
    end
    return best_solution
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    test_PDPInstance(
        ;
        path_dir="instances",
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend,
    )
end