#!/usr/bin/env julia
include("paths.jl")
using Printf, Logging, DelimitedFiles, JSON, Random, StatsBase

############# PROBLEM - VIENNA SEIDL RALLY #############

# https://www.seidltour.at/

N_DISTRICTS = 23
N_BARS = N_DISTRICTS * 3

function get_random_bar(district::Int64)
    idx = district * 3
    locs = idx-2:idx
    return rand(locs)
end

function get_random_trail(n_districts::Int64, n_bars::Int64)
    random_trail = Vector{Int64}()
    open_locs = collect(1:n_bars)
    for _ in 1:n_districts
        loc_choice = rand(open_locs)
        trail_step_update!(random_trail, open_locs, loc_choice)
    end
    return random_trail
end

function get_trail_transport(choice_matrix::Matrix{Bool}, trail::Vector{Int64})
    transport = Vector{String}()
    for k in 1:length(trail)-1
        push!(transport, choice_matrix[trail[k], trail[k+1]] ? "walking" : "public-transport")
    end
    return transport
end

function get_trail_cost(cost_matrix::Matrix{Float64}, trail::Vector{Int64})
    trail_cost = 0
    for i in 1:length(trail)-1
        trail_cost += cost_matrix[trail[i], trail[i+1]]
    end
    return trail_cost
end

############# ALGORITHM - ANT COLONY SYSTEM #############

# https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
# https://dataworldblog.blogspot.com/2017/06/ant-colony-optimization-part-1.html
# https://cleveralgorithms.com/nature-inspired/swarm/ant_colony_system.html
# https://github.com/nishnash54/TSP_ACO

function state_trans_prob(
    ant_curr_loc::Int64,
    ant_open_locs::Vector{Int64},
    phero_level::Matrix{Float64},
    attractiveness_level::Matrix{Float64},
    alpha::Float64,
    beta::Float64,
)
    i = ant_curr_loc
    total_level_norm = 0
    for l in ant_open_locs
        total_level_norm += (phero_level[i, l]^alpha) * (attractiveness_level[i, l]^beta)
    end
    ant_trans_probs = zeros(length(ant_open_locs))
    for (idx, j) in enumerate(ant_open_locs)
        ant_trans_probs[idx] = (phero_level[i, j]^alpha) * (attractiveness_level[i, j]^beta) / total_level_norm
    end
    return ant_trans_probs
end

function trans_greedy(ant_trans_probs::Vector{Float64})
    return argmax(ant_trans_probs)
end

function trans_prob(ant_trans_probs::Vector{Float64})
    return sample(CartesianIndices(ant_trans_probs), weights(ant_trans_probs))[1]
end

function update_pheromones_local!(
    ant_trail::Vector{Int64},
    phero_level::Matrix{Float64},
    local_phero_decay::Float64,
    phero_init::Float64,
)
    for k in 1:length(ant_trail)-1
        i, j = ant_trail[k], ant_trail[k+1]
        phero_update = (1 - local_phero_decay) * phero_level[i, j] + local_phero_decay * phero_init
        phero_level[i, j] = phero_update
        phero_level[j, i] = phero_update
    end
end

function update_pheromones_global!(
    best_trail::Vector{Int64},
    best_cost::Float64,
    phero_level::Matrix{Float64},
    phero_decay::Float64,
)
    n_bars = size(phero_level)[1]
    for i in 1:n_bars
        for j in 1:n_bars
            position_i = findfirst(==(i), best_trail)
            ij_in_trail = position_i != nothing && position_i + 1 <= length(best_trail) && (j == best_trail[position_i+1])
            phero_delta = ij_in_trail ? 1 / best_cost : 0
            phero_update = (1 - phero_decay) * phero_level[i, j] + phero_decay * phero_delta
            phero_level[i, j] = phero_update
        end
    end
end

function trail_step_update!(
    ant_trail::Vector{Int64},
    ant_open_locs::Vector{Int64},
    loc_next::Int64
)
    # add new district location to trail
    push!(ant_trail, loc_next)
    # remove all related district locations from open
    loc_idx = loc_next % 3
    if loc_idx == 1
        shift_ids = [0, 1, 2]
    elseif loc_idx == 2
        shift_ids = [-1, 0, +1]
    else # loc_idx == 0
        shift_ids = [-2, -1, 0]
    end
    for del_idx in (loc_next .+ shift_ids)
        deleteat!(ant_open_locs, findfirst(==(del_idx), ant_open_locs))
    end
end

############# MAIN #############

function vsr_ant_colony_algorithm(n_districts::Int64, n_bars::Int64, cost_matrix::Matrix{Float64}, choice_matrix::Matrix{Bool}, m_colony::Int64, n_iter::Int64)
    # initial random trail
    trail = get_random_trail(n_districts, n_bars)
    cost = get_trail_cost(cost_matrix, trail)
    # desirability of state transition // a priori
    attractiveness_level = 1 ./ cost_matrix
    # amount of pheromones // a posteriori
    phero_init = 1 / n_districts * cost
    phero_level = ones(n_bars, n_bars) * phero_init
    # pheromone evaporation coefficients
    local_phero_decay = 0.1
    phero_decay = 0.1
    # state transition coefficients
    beta = 3.0
    alpha = 1.0
    greed_factor = 0.9
    # problem specific variables
    final_district = 1

    best_trail = trail
    best_cost = cost
    n_tries = 0
    n_iter_cost = Vector{Float64}()
    while n_tries < n_iter # for n tries
        colony_trails = [Vector{Int64}() for _ in 1:m_colony]
        colony_open_locs = [collect(1:n_bars) for _ in 1:m_colony]
        for k in 1:m_colony # for each ant
            colony_final_loc = get_random_bar(final_district)
            ant_open_locs = colony_open_locs[k]
            ant_trail = colony_trails[k]
            trail_step_update!(ant_trail, ant_open_locs, colony_final_loc)
            while length(ant_open_locs) > 0
                ant_trans_probs = state_trans_prob(
                    ant_trail[end],
                    ant_open_locs,
                    phero_level,
                    attractiveness_level,
                    alpha,
                    beta,
                )
                greedy_select = rand() <= greed_factor
                ant_next_loc_idx = greedy_select ? trans_greedy(ant_trans_probs) : trans_prob(ant_trans_probs)
                ant_next_loc = ant_open_locs[ant_next_loc_idx]
                trail_step_update!(ant_trail, ant_open_locs, ant_next_loc)
            end
            update_pheromones_local!(ant_trail, phero_level, local_phero_decay, phero_init)
            ant_cost = get_trail_cost(cost_matrix, ant_trail)
            if ant_cost < best_cost
                best_cost = ant_cost
                best_trail = ant_trail
            end
        end
        update_pheromones_global!(best_trail, best_cost, phero_level, phero_decay)
        push!(n_iter_cost, best_cost)
        @info "Iteration $(n_tries+1) yielded best score: $(@sprintf("%.2f", best_cost))."
        n_tries += 1
    end
    best_trail = reverse(best_trail) # move final district to end
    best_districts = ceil.(Int64, best_trail ./ 3)
    best_transport = get_trail_transport(choice_matrix, best_trail)
    @info "Ant colony system returned location path: $best_trail."
    @info "Ant colony system returned district path: $best_districts."
    @info "Ant colony system returned transport choices: $best_transport."
    @info "Ant colony system returned solution cost: $(@sprintf("%.2f", best_cost))."
    return (best_districts, best_trail, best_transport, best_cost, n_iter_cost)
end

function run_vsr_ant_colony_algorithm(; m_colony::Int64=200, n_iter::Int64=3000)
    Random.seed!(1234)
    duration_matrix_walk = readdlm(WALK_DUR_PATH, Float64)
    duration_matrix_public = readdlm(PUBLIC_DUR_PATH, Float64)
    cost_matrix = min.(duration_matrix_walk, duration_matrix_public)
    choice_matrix = Matrix{Bool}(duration_matrix_walk .< duration_matrix_public)
    return vsr_ant_colony_algorithm(N_DISTRICTS, N_BARS, cost_matrix, choice_matrix, m_colony, n_iter)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    println("Running ACS in Standalone Mode...")
    run_vsr_ant_colony_algorithm()
end