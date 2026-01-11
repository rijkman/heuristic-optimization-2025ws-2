using ArgParse
using Random

include("acs.jl")
include("greedy_random.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n-iterations"
            arg_type = Int
            required = true
        "--m-colony"
            arg_type = Int
            required = true
        "--l-phero-decay"
            arg_type = Float64
            required = true
        "--phero-decay"
            arg_type = Float64
            required = true
        "--beta"
            arg_type = Float64
            required = true
        "--alpha"
            arg_type = Float64
            required = true
        "--factor-greed"
            arg_type = Float64
            required = true
        "--seed"
            arg_type = Int64
            required = true
        "--instance"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function parse_instance(path_in::String)
    open(path_in, "r") do f
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

function main()
    args = parse_commandline()

    Random.seed!(args["seed"])    

    instance_data = args["instance"]
    instance = parse_instance(instance_data)
    (_, _, _, init_score) = greedy_heuristic_one_extend_random(instance, verbose = false)

    start_time = time()

    (iter_score, iter_n, best_solution, best_score) = delta_ant_colony_system(
        instance;
        n_iter = args["n-iterations"],
        m_colony = args["m-colony"],
        local_phero_decay = args["l-phero-decay"],
        phero_decay = args["phero-decay"],
        beta = args["beta"],
        alpha = args["alpha"],
        factor_greed = args["factor-greed"],
        score = init_score,
        verbose = false
    )

    print("$best_score $(time()-start_time)\n")
end

main()