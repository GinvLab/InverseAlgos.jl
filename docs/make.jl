using Documenter, InverseAlgos

makedocs(modules = [InverseAlgos],
         repo=Remotes.GitHub("GinvLab","InverseAlgos.jl"), 
         sitename="InverseAlgos.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
             "MCSamplers" => "mcsamplers.md",
             "Optimizers" => "optimizers.md",
             "KronLinInv" => "kronlininv.md"
         ],
         warnonly=true
         )

deploydocs(
    repo="github.com/GinvLab/InverseAlgos.jl.git", 
    devbranch = "main",
    deploy_config = Documenter.GitHubActions(),
    branch = "gh-pages"
)


