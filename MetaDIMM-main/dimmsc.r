tic()
result <- DIMMSC(data=t(ad$X), K=8, method_cluster_initial="kmeans", method_alpha_initial="Ronning", maxiter=100, tol=1e-4, lik.tol=1e-2)
toc()
plot_tsne_clusters(data=t(ad$X), cluster=result$mem)
