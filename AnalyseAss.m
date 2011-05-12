function [ass, count, present] = AnalyseAss( correct, MC, fr)
%ANALYSEASS Compare associations with correct values

global Par;

ass = cell(Par.NumTgts, 1);
count = zeros(Par.NumTgts, fr);
present = zeros(Par.NumTgts, fr);

for j = 1:MC.particles{1}.N
    
    ass{j} = zeros(Par.NumIt, fr);
    
    for t = 1:fr
        
        ass{j}(:, t) = cellfun(@(x) x.tracks{j}.GetAssoc(t), MC.particles);
        pres_array = cellfun(@(x) x.tracks{j}.Present(t), MC.particles);
        present(j, t) = present(j, t) + sum(pres_array);
        
        count(j, t) = sum(ass{j}(:, t)==correct(t, j));
        
    end
    
end

figure, hold on
for j = 1:Par.NumTgts
    plot(count(j, :), 'color', [0, rand, rand])
end

end