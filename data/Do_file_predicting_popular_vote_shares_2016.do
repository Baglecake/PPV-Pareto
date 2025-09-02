clear 
clear matrix
clear mata
set maxvar 30000
use "/[insert path]/ANES_data_predicting_popular_vote_shares_2016.dta"


*recode variables*
 
*vote choice
gen vote_Trump = V161031
replace vote_Trump =1 if  V161031 == 2
replace vote_Trump =0 if  V161031 == 1 | V161031 == 3 | V161031 == 4 | V161031 == 5
replace vote_Trump =. if  V161031 == 6 |  V161031 ==7 |  V161031 == 8 |  V161031 == -9 |  V161031 == -8 | V161031 == -1


*ideology*
gen lib_cons = V161126
recode lib_cons (-9/-8=.) (99=.)

 
*Disapproval of economy during last year*
gen Disapp_economy =  V161083
recode Disapp_economy (-9/-8=.)
  
*Economic situation
gen pers_econ_worse = V161110
recode pers_econ_worse (-9/-8=.) 
  
*Distrust in government
gen distrust_gov = V161215
recode distrust_gov (-9/-8=.) 
 
*Age*
gen age = V161267
recode age (-9/-8=.) 
 
*Gender*
gen female = V161342
recode female (2=1) (1=0) (3=0) (-9=.)
 
*Education*
gen edu = V161270
recode edu (-9=.) (90/95=.)
 
*Ethnicity*
gen white_nonhispanic = V161310x
recode white_nonhispanic (-2=.) (2/6=0)
 
*state*
gen state = V161015b
recode state (-1=.) 


**variables' normalisation (0 to 1)**
 
gen lib_cons_norm = (lib_cons - 1) / 6
gen Disapp_economy_norm = Disapp_economy - 1
gen pers_econ_worse_norm = (pers_econ_worse - 1) / 4
gen Distrust_gov_norm = (distrust_gov  - 1)/4
gen age_norm = (age - 18) / (90-18)
gen edu_norm = (edu - 1) / 15



*logit model - vote intention (Table A1)*

logit vote_Trump lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state , vce(robust)
logit, or

predict yhat_pre if vote_Trump !=. & lib_cons_norm !=. & Disapp_economy_norm !=. & pers_econ_worse_norm !=. & Distrust_gov_norm !=. & age_norm !=. & edu_norm !=. & female !=. & white_nonhispanic !=. & state !=.

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
gen predicted_Trump_pre_vote2 = 0
recode predicted_Trump_pre_vote2 (0=1) if yhat_pre > .5
replace predicted_Trump_pre_vote2 = . if yhat_pre == . 


***b) replication with average predicted probability
svyset [pweight=V160101]
svy: mean yhat_pre
gen predicted_Trump_pre_vote3 = 0
recode predicted_Trump_pre_vote3 (0=1) if yhat_pre > .4291535 
replace predicted_Trump_pre_vote3 = . if yhat_pre == . 



*replication with Bayesian logistic model (Table A2)*


bayesmh vote_Trump lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state , ///
    likelihood(logistic) ///
    prior({lib_cons_norm}, normal(0, 10)) ///
    prior({Disapp_economy_norm}, normal(0, 10)) ///
    prior({Distrust_gov_norm}, normal(0, 10)) ///
    prior({age_norm}, normal(0, 10)) ///
    prior({edu_norm}, normal(0, 10)) ///
    prior({i.white_nonhispanic}, normal(0, 10)) ///
    prior({pers_econ_worse_norm}, normal(0, 10)) ///
    prior({i.female}, normal(0, 10)) ///
	prior({i.state}, normal(0, 10)) ///
	prior({_cons}, normal(0, 10)) ///
    mcmcsize(1000) burnin(5000) rseed(12345) saving(mcmc2016, replace)
	
	bayespredict vote_Trump_p , mean rseed(12345)

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
	gen simul_Trump_vote = 0
	recode simul_Trump_vote (0=1) if vote_Trump_p >  .5
	replace simul_Trump_vote = . if vote_Trump_p == . 

***b) replication with average predicted probability
	svy: mean vote_Trump_p
	gen simul_Trump_vote2 = 0
	recode simul_Trump_vote2 (0=1) if vote_Trump_p  >  .4273632
	replace simul_Trump_vote2 = . if vote_Trump_p  == . 

***c) Aggregation and comparison of predicted votes according to various strategies (Table 1)
mean vote_Trump
svy: mean vote_Trump
svy: mean predicted_Trump_pre_vote2  
svy: mean predicted_Trump_pre_vote3
svy: mean simul_Trump_vote 
svy: mean simul_Trump_vote2
