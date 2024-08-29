# Statistic Analysis Results
This folder contains detailed additional statistical analysis and plots for "Summary Findings" in our paper. 

## Important values for AI 

[This figure](https://github.com/JuneHou/XAI_MOR_Survey/blob/main/Statistic%20Analysis/Feature_Experience.png) shows most  important values for AI selected by participants from among a list of 13 values adapted from [Jakesch et al. (2022)](https://dl.acm.org/doi/fullHtml/10.1145/3531146.3533097), split by whether the participant has
had experience with AI.. Among those without AI experience (the majority), the most important values highlighted were Safety (24), Privacy (20), Performance (17), Accountability (13), Human Autonomy (12), and Transparency (11). Among those with AI experimence, the most important values selected were Safety (6), Performance (5), Beneficence (5), Privacy (4), Accountability (3), Human Autonomy (3), Transparency (3), and Dignity (3).

## Comfort level for decision-making with admission note 

[This figure](Statistic Analysis/comfort_edu.png) displays results categorized by the participant’s highest level of education. Before presenting the XAI methods, we show participants an unchanged version of a patient admission note and asked participants to report their comfort level, on a five-point scale, for predicting the mortality outcome using the note. A large majority of participants reported being slightly or very comfortable with the decision-making task (n=22). Whereas only 4 participants reported being slightly or very uncomfortable with the task.

## Amount of information provided by each XAI method 

When evaluating clinician perception of each XAI method, we asked them to assess the appropriateness of the amount of information provided on a five-point scale, ranging from much less information desired to much more information desired. Figure 12 presents the results. For LIME and attention- based highlights, we highlighted the top 20% and 30% of important features identified by the methods respectively. For LIME, 19 participants indicated wanting much more or slightly more highlights. For Attention, 21 participants indicated wanting much more or slightly more highlights. A small number of participants (5 for LIME and 5 for Attention) indicated wanting fewer highlights. For similar patient retrieval, 12 participants indicated a desire for much or slightly more notes for comparison, while 4 wanted fewer. For rationales, a large number of participants (19) were satisfied with the number of rationales (3), while 13
participants indicated wanting more. Although the preference for the amount of information provided among the four methods is not consistent, participants generally favored more rather than fewer explanations.

## Ranking XAI methods 

After being shown all four XAI methods in the survey, we asked participants to rank them in terms of understandability, how reasonable they are, as well as overall preference. Summary rankings are shown in Figure 13. Free-text rationales were ranked as most understandable (16 people ranked first), most reasonable (17 people ranked first), and most preferred overall (15 people ranked first). LIME was also highly ranked, with 9 people ranking it first on understandability, 8 first on reasonableness, and 12 first overall.
