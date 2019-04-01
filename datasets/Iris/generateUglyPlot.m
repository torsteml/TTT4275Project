load('PlotOFAlphaSweep.mat');
alpha = [0.00001:1/10000:0.1];

plot(alpha,ACCs);
ylabel("Nøyaktighet");
xlabel("Alpha")
%legend(["10 iterasjoner","100 iterasjoner", "1'000 iterasjoner","10'000 iterasjoner","100'000 iterasjoner"]);