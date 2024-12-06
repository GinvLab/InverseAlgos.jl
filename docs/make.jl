using Documenter, InverseAlgos

makedocs(modules = [InverseAlgos],
         repo=Remotes.GitHub("GinvLab","InverseAlgos.jl"), 
         sitename="InverseAlgos.jl",
         authors = "Andrea Zunino",
         format = Documenter.HTML(prettyurls=get(ENV,"CI",nothing)=="true"),
         pages = [
             "Home" => "index.md",
             "MCSamplers" => "mcsamplers.md",
             "Optimizers" => "optimizers.md"
         ],
         #warnonly=true
         )

deploydocs(
    repo="github.com/GinvLab/MCsamplers.jl.git", 
    devbranch = "main",
    deploy_config = Documenter.GitLab(),
    branch = "gh-pages"
)


