clear
clear matrix
clear mata
set maxvar 30000
use "/[insert path]/ANES_data_predicting_popular_vote_shares_2020.dta"
 

*recode variables*
 
 *vote choice
gen vote_Trump = V201033
replace vote_Trump =1 if  V201033 == 2
replace vote_Trump =0 if  V201033 == 1 | V201033 == 3 | V201033 == 4 | V201033 == 5
replace vote_Trump =. if  V201033 ==-8 |  V201033 ==-9 |  V201033 ==-1 | V201033 ==11 |  V201033 ==12


*ideology*
gen lib_cons = V201200
recode lib_cons (-9/-8=.) (99=.)

 
*Disapproval of economy during last year*
gen Disapp_economy = V201327x
recode Disapp_economy (-2=.)
  
*Economic situation
gen pers_econ_worse = V201502
recode pers_econ_worse (-9=.) 

 
*Distrust in government
gen Distrust_gov = V201233
recode Distrust_gov (-9/-8=.)
 
*Age*
gen age = V201507x
recode age (-9=.) 
 
*Gender*
gen female = V201600
recode female (2=1) (1=0) (-9=.)
 
*Education*
gen edu = V201511x
recode edu (-9/-2=.)
  
*Ethnicity*
gen white_nonhispanic = V201549x
recode white_nonhispanic (-9/-8=.) (2/6=0)
 
*state*
gen state = V201014b
recode state (-9/-1=.) (86=.)



*variables' normalisation (0 to 1)*
 
gen lib_cons_norm = (lib_cons - 1) / 6
gen Disapp_economy_norm = (Disapp_economy - 1) / 4
gen pers_econ_worse_norm = (pers_econ_worse - 1) / 4
gen age_norm = (age - 18) / (80-18)
gen edu_norm = (edu - 1) / 4
gen Distrust_gov_norm = (Distrust_gov - 1) / 4
 

*logit model - vote intention (Table A1)*

logit vote_Trump lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state , vce(robust)
logit, or

predict yhat_pre if vote_Trump !=. & lib_cons_norm !=. & Disapp_economy_norm !=. & pers_econ_worse_norm !=. & Distrust_gov_norm !=. & age_norm !=. & edu_norm !=. & female !=. & white_nonhispanic !=. & state !=.

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
gen predicted_Trump_pre_vote2 = 0
recode predicted_Trump_pre_vote2 (0=1) if yhat_pre > .5
replace predicted_Trump_pre_vote2 = . if yhat_pre == . 


***b) replication with average predicted probability
svyset [pweight=V200010a]
svy: mean yhat_pre
gen predicted_Trump_pre_vote3 = 0
recode predicted_Trump_pre_vote3 (0=1) if yhat_pre > .4435452 
replace predicted_Trump_pre_vote3 = . if yhat_pre == . 



*replication with Bayesian logistic model (Table A2)*


bayesmh vote_Trump lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state  , ///
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
    mcmcsize(1000) burnin(5000) rseed(12345) saving(mcmc2020, replace)
	
	bayespredict vote_Trump_p , mean rseed(12345)

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
	gen simul_Trump_vote = 0
	recode simul_Trump_vote (0=1) if vote_Trump_p >  .5
	replace simul_Trump_vote = . if vote_Trump_p == . 

***b) replication with average predicted probability
	svy: mean vote_Trump_p
	gen simul_Trump_vote2 = 0
	recode simul_Trump_vote2 (0=1) if vote_Trump_p  >  .4428464 
	replace simul_Trump_vote2 = . if vote_Trump_p  == . 

***c) Aggregation and comparison of predicted votes for Trump according to various strategies (Table 1)
mean vote_Trump
svy: mean vote_Trump
svy: mean predicted_Trump_pre_vote2  
svy: mean predicted_Trump_pre_vote3
svy: mean simul_Trump_vote 
svy: mean simul_Trump_vote2
