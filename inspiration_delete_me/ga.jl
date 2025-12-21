#!/usr/bin/env julia
include("paths.jl")
using CSV, DataFrames, Random, StatsBase

get_key(type, metric) = "$type-$metric"
array_size_equal(arrays) = isempty(arrays) || all(size(arr) == size(arrays[1]) for arr in arrays)

function parse_datafile(path)
    lines = readlines(path)
    nrows = length(lines)
    ncols = length(split(lines[1]))
    A = Array{Float64}(undef, nrows, ncols)
    for (i, line) in enumerate(lines)
        A[i, :] = parse.(Float64, split(line))
    end
    return A
end

function load_location_metrics(route_type, route_metric)
    Dict(
        get_key(t, m) => parse_datafile(joinpath(ROUTING_PATH, "location_$(t)_$(m).txt"))
        for t in route_type, m in route_metric
    )
end

function calculate_decision_matrix(location_data, route_type, metric)
    arrays = [location_data[get_key(t, metric)] for t in route_type]
    isempty(arrays) && error("Selection must produce at least 1 element")
    array_size_equal(arrays) || error("Unequal array sizes")
    A = cat(arrays...; dims=3)
    ci = argmin(A, dims=3)
    decision_matrix = dropdims(getindex.(ci, 3); dims=3)
    mixed_cost_3d = minimum(A, dims=3)
    mixed_cost = dropdims(mixed_cost_3d; dims=3)
    return decision_matrix, mixed_cost
end

# =======================================================================================================

struct SeidlRallyDescription
    metric_matrix::Matrix{Float64}
    district_of_pub::Vector{Int}
    district_pubs::Vector{Vector{Int}} # Index = District ID, Value = List of Pub IDs in that district
    end_district::Int
end

mutable struct Individual
    district_order::Vector{Int} # The sequence of visiting districts
    pub_choices::Vector{Int} # The specific pub chosen for each district. Index is District ID.
    fitness::Float64
end

# --- decoding and fitness ---

function decode_route(ind::Individual)::Vector{Int}
    # Route is: For every district in the district_order, visit the specifically chosen pub
    return [ind.pub_choices[d] for d in ind.district_order]
end

function route_cost(route::Vector{Int}, T::Matrix{Float64})
    total = 0.0
    @inbounds for i in 1:(length(route)-1)
        total += T[route[i], route[i+1]]
    end
    return total
end

function evaluate!(ind::Individual, problem::SeidlRallyDescription)
    route = decode_route(ind)
    ind.fitness = route_cost(route, problem.metric_matrix)
    return ind
end

# --- initialization ---

function random_individual(problem::SeidlRallyDescription, rng::AbstractRNG=Random.default_rng())
    D = length(problem.district_pubs)
    end_d = problem.end_district

    districts = collect(1:D)
    deleteat!(districts, findfirst(==(end_d), districts))
    shuffle!(rng, districts)
    push!(districts, end_d)

    pub_choices = Vector{Int}(undef, D)
    for d in 1:D
        pub_choices[d] = rand(rng, problem.district_pubs[d])
    end

    ind = Individual(districts, pub_choices, Inf)
    return evaluate!(ind, problem)
end

# --- crossover ---

# Order Crossover (OX1) for the permutation of districts
function ox_crossover_districts(p1::Vector{Int}, p2::Vector{Int}, end_district::Int, rng::AbstractRNG)
    D = length(p1)
    @assert p1[end] == end_district
    @assert p2[end] == end_district

    if D <= 2
        return copy(p1)
    end

    child = fill(-1, D)

    cut1 = rand(rng, 1:(D-2))
    cut2 = rand(rng, (cut1+1):(D-1))

    child[cut1:cut2] .= p1[cut1:cut2]

    pos = 1
    for gene in p2[1:(D-1)]
        if gene == end_district || (gene in child)
            continue
        end
        while child[pos] != -1
            pos += 1
        end
        child[pos] = gene
    end

    child[end] = end_district
    return child
end

# Uniform Crossover for Pub Choices
function crossover_pub_choices(p1::Individual, p2::Individual, rng::AbstractRNG)
    D = length(p1.pub_choices)
    child_choices = Vector{Int}(undef, D)
    for d in 1:D
        if rand(rng) < 0.5
            child_choices[d] = p1.pub_choices[d]
        else
            child_choices[d] = p2.pub_choices[d]
        end
    end
    return child_choices
end

function crossover(p1::Individual, p2::Individual, problem::SeidlRallyDescription, rng::AbstractRNG; pcross::Float64=0.9)
    if rand(rng) > pcross
        return (deepcopy(p1), deepcopy(p2))
    end

    end_d = problem.end_district

    c1_districts = ox_crossover_districts(p1.district_order, p2.district_order, end_d, rng)
    c2_districts = ox_crossover_districts(p2.district_order, p1.district_order, end_d, rng)

    c1_pubs = crossover_pub_choices(p1, p2, rng)
    c2_pubs = crossover_pub_choices(p2, p1, rng)

    c1 = Individual(c1_districts, c1_pubs, Inf)
    c2 = Individual(c2_districts, c2_pubs, Inf)
    return c1, c2
end

# --- mutation ---

function mutate_districts!(ind::Individual, problem::SeidlRallyDescription, rng::AbstractRNG; pmut::Float64=0.1)
    D = length(ind.district_order)
    if D > 2 && rand(rng) < pmut
        i, j = sort(rand(rng, 1:(D-1), 2))
        reverse!(ind.district_order, i, j)
    end
end

function mutate_pub_choices!(ind::Individual, problem::SeidlRallyDescription, rng::AbstractRNG; pmut::Float64=0.2)
    # With probability pmut, pick a random district and change the selected pub
    if rand(rng) < pmut
        D = length(ind.pub_choices)
        d = rand(rng, 1:D)

        available_pubs = problem.district_pubs[d]
        current_pub = ind.pub_choices[d]

        if length(available_pubs) > 1
            new_pub = rand(rng, filter(x -> x != current_pub, available_pubs))
            ind.pub_choices[d] = new_pub
        end
    end
end

function mutate!(ind::Individual, problem::SeidlRallyDescription, rng::AbstractRNG; pdistrict::Float64=0.1, ppub::Float64=0.1)
    mutate_districts!(ind, problem, rng; pmut=pdistrict)
    mutate_pub_choices!(ind, problem, rng; pmut=ppub)
end

# --- selection, reproduction, evolution ---

function tournament_select(pop::Vector{Individual}, k::Int, rng::AbstractRNG)
    best = rand(rng, pop)
    for _ in 2:k
        cand = rand(rng, pop)
        if cand.fitness < best.fitness
            best = cand
        end
    end
    return best
end

function reproduce(p1::Individual, p2::Individual, problem::SeidlRallyDescription, rng::AbstractRNG)
    c1, c2 = crossover(p1, p2, problem, rng)
    mutate!(c1, problem, rng)
    mutate!(c2, problem, rng)
    evaluate!(c1, problem)
    evaluate!(c2, problem)
    return c1, c2
end

function evolve(problem::SeidlRallyDescription;
    popsize::Int=100,
    generations::Int=500,
    tournament_k::Int=3,
    rng::AbstractRNG=Random.default_rng())

    population = [random_individual(problem, rng) for _ in 1:popsize]

    history = Float64[]
    for gen in 1:generations
        sort!(population, by=ind -> ind.fitness)

        best_gen = population[1].fitness
        push!(history, best_gen)

        # Elitism: Keep the best one
        new_pop = Vector{Individual}()
        push!(new_pop, deepcopy(population[1]))

        while length(new_pop) < popsize
            p1 = tournament_select(population, tournament_k, rng)
            p2 = tournament_select(population, tournament_k, rng)
            c1, c2 = reproduce(p1, p2, problem, rng)

            push!(new_pop, c1)
            if length(new_pop) < popsize
                push!(new_pop, c2)
            end
        end

        population = new_pop
        #if gen % 50 == 0
        println("Gen $gen, best cost = $(best_gen)")
        #end
    end

    sort!(population, by=ind -> ind.fitness)
    return population[1], history
end

# --- problem builder ---

function build_problem(T::Matrix{Float64}, district_of_pub::Vector{Int}, end_district::Int)
    N = length(district_of_pub)
    @assert size(T, 1) == N && size(T, 2) == N

    D = maximum(district_of_pub)
    district_pubs = [Int[] for _ in 1:D]
    for pub in 1:N
        d = district_of_pub[pub]
        push!(district_pubs[d], pub)
    end

    return SeidlRallyDescription(T, district_of_pub, district_pubs, end_district)
end


function decode_route_modes(route::Vector{Int}, decision_matrix::Matrix{Int}, route_profile::Vector{String})
    modes = String[]
    for k in 1:(length(route)-1)
        i = route[k]
        j = route[k+1]
        idx = decision_matrix[i, j]
        push!(modes, route_profile[idx])
    end
    return modes
end

function run_vsr_genetic_algorithm(; popsize::Int64=2000, generations::Int64=1000)
    ROUTE_TYPE = ["walk", "public"]
    ROUTE_PROFILE = ["walking", "public-transport"]
    ROUTE_METRIC = ["distance", "duration"]

    location_data = load_location_metrics(ROUTE_TYPE, ROUTE_METRIC)
    decision_matrix, mixed_cost = calculate_decision_matrix(location_data, ROUTE_TYPE, "duration")
    locations = CSV.read(BAR_PATH, DataFrame)
    locations.district = (locations.postcode .- 1000) .÷ 10
    district_vec = collect(locations.district)

    problem = build_problem(mixed_cost, district_vec, 1)
    best, cost_history = evolve(problem, popsize=popsize, generations=generations, rng=MersenneTwister(42))

    best_route = decode_route(best)
    best_districts = ceil.(Int64, best_route ./ 3)
    best_transport = decode_route_modes(best_route, decision_matrix, ROUTE_PROFILE)
    best_cost = best.fitness

    println("GA returned solution cost: $best_cost")
    println("GA returned district path: $best_districts")
    println("GA returned district pub path: $best_route")

    return best_districts, best_route, best_transport, best_cost, cost_history
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running GA in Standalone Mode...")
    run_vsr_genetic_algorithm()
end