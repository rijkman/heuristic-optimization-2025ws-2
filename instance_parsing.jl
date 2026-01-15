#!/usr/bin/env julia
using ArgParse, Printf, Logging, CPUTime, JSON, CSV, DataStructures, Random, IterTools, Statistics, StatsBase, Combinatorics, Plots
Plots.default(show=true)

struct PDPInstance
    n_requests::Int64                               # amount of requests to be made
    n_vehicles::Int64                               # amount of vehicles available
    capacity::Int64                                 # max capacity of one vehicle
    γ::Int64                                        # min amount of requests to be fulfilled
    ρ::Float32                                      # weighting param that controls objective f trade-off
    demands::Vector{Int64}                          # demands; amount of goods needed per request
    requests::Vector{Tuple{Int64,Int64,Bool,Int64}} # pickup index, dropoff index, pickup flag, load/demand
    coords_depot::Tuple{Int64,Int64}                # coords of the depot
    coords_pickups::Vector{Tuple{Int64,Int64}}      # coordinates of pickup points
    coords_dropoffs::Vector{Tuple{Int64,Int64}}     # coordinates of dropoff points
    locations::Vector{Tuple{Int64,Int64}}           # coordinates of pickups, dropoffs and depot
    distance_matrix::Matrix{Int64}                  # distances between all locations
end

PDPSolutionVector = Vector{Vector{Int64}}
PDPSwap = Vector{NTuple{4,Int64}}
PDPNeighborhood = BinaryMinHeap{Tuple{Float64,PDPSwap}}

#region ############## I/O PARSING ##############

BASE_DIR = "data"
INSTANCE_DIR = "instances"

function read_PDPInstance(path_full::String)
    @info "Reading new instance."
    open(path_full, "r") do f
        # read in params
        params = split(readline(f), " ")
        n_requests, n_vehicles, capacity, γ = parse.(Int64, (params[1:4]))
        ρ = parse(Float32, String(params[5]))
        # read in demands
        readline(f) # consume demands header
        demands = parse.(Int64, split(readline(f), " "))
        # read in request locations
        readline(f) # consume rq locs header
        requests::Vector{Tuple{Int64,Int64,Bool,Int64}} = []
        coords_depot = (parse.(Int64, split(readline(f), " "))...,)
        coords_pickups::Vector{Tuple{Int64,Int64}} = []
        coords_dropoffs::Vector{Tuple{Int64,Int64}} = []
        for i in 1:n_requests
            current_line = readline(f)
            l_x, l_y = parse.(Int64, split(current_line, " "))
            push!(requests, (i, i + n_requests, true, demands[i]))
            push!(coords_pickups, (l_x, l_y))
        end
        for i in 1:n_requests
            current_line = readline(f)
            l_x, l_y = parse.(Int64, split(current_line, " "))
            push!(requests, (i, i + n_requests, false, demands[i]))
            push!(coords_dropoffs, (l_x, l_y))
        end
        locations = vcat(coords_pickups, coords_dropoffs, coords_depot)
        distance_matrix = dist_matrix(locations)
        return PDPInstance(
            n_requests,
            n_vehicles,
            capacity,
            γ,
            ρ,
            demands,
            requests,
            coords_depot,
            coords_pickups,
            coords_dropoffs,
            locations,
            distance_matrix,
        )
    end
end

function solve_PDPInstance(
    ;
    instance_size::String,
    instance_type::String,
    instance_name::String,
    algorithm::Function,
    args...,
)
    path_in = joinpath(pwd(), BASE_DIR, INSTANCE_DIR, instance_size, instance_type, instance_name)
    instance = read_PDPInstance(path_in)
    return algorithm(instance; args...)
end

function test_PDPInstance(
    ;
    instance_size::String,
    instance_type::String,
    instance_name::String,
    algorithm::Union{Function,Nothing}=nothing,
    args...,
)
    path_in = joinpath(pwd(), BASE_DIR, INSTANCE_DIR, instance_size, instance_type, instance_name)
    instance = read_PDPInstance(path_in)
    if algorithm == nothing
        visualize(instance)
    else
        _, _, solution, score = algorithm(instance; args...)
        visualize(instance, solution, score)
    end
    readline()
end

#endregion

#region ############## DISTANCE ##############

function dist(P1::Tuple{Int64,Int64}, P2::Tuple{Int64,Int64})
    x1, y1 = P1
    x2, y2 = P2
    return ceil(Int64, sqrt((x1 - x2)^2 + (y1 - y2)^2))
end

function dist_matrix(locations::Vector{Tuple{Int64,Int64}})::Matrix{Float64}
    n = length(locations)
    dist_matrix = zeros(n, n)
    for i in 1:n
        for j in 1:n
            dist_matrix[i, j] = dist(locations[i], locations[j])
        end
    end
    return dist_matrix
end

#endregion

#region ############## VISUALIZATION ##############

function visualize(instance::PDPInstance)
    p = plot(instance.coords_depot, seriestype=:scatter, markershape=:star5, markersize=15, size=(900, 900), label="depot")
    plot!(instance.coords_pickups, seriestype=:scatter, markershape=:circle, markersize=5, label="pickups")
    plot!(instance.coords_dropoffs, seriestype=:scatter, markershape=:square, markersize=5, label="dropoffs")
    coords_dirs = collect.(instance.coords_dropoffs) .- collect.(instance.coords_pickups)
    quiver!(instance.coords_pickups, quiver=(first.(coords_dirs), last.(coords_dirs)), color=:black, linealpha=0.3)
    return p
end

function visualize(instance::PDPInstance, solution::PDPSolutionVector, score::Float64)
    p = visualize(instance)
    annotate!(xlims(p)[2], ylims(p)[2], text("Score: $(@sprintf("%.2f", score))", :right))
    for k in 1:instance.n_vehicles
        if !isempty(solution[k])
            solution_k = [instance.coords_depot; instance.locations[solution[k]]; instance.coords_depot]
            plot!(solution_k, seriestype=:path, linewidth=:2, label="Car $k")
        end
    end
    return p
end

#endregion

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    test_PDPInstance(
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
    )
end