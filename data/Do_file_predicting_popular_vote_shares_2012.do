clear
clear matrix
clear mata
set maxvar 30000 
use "/[insert path]/ANES_data_predicting_popular_vote_shares_2012.dta"


*recode variables*
 
 *vote choice
gen vote_Romney = prevote_intpreswho
replace vote_Romney =1 if  prevote_intpreswho == 2
replace vote_Romney =0 if  prevote_intpreswho == 1 | prevote_intpreswho == 5
replace vote_Romney =. if  prevote_intpreswho == -9 |  prevote_intpreswho == -8 |  prevote_intpreswho == -1 

*ideology*
gen lib_cons = libcpre_self
recode lib_cons (-9/-8=.) (-2=.)

 
*Disapproval of economy during last year*
gen Disapp_economy =  presapp_econ
recode Disapp_economy (-9/-8=.)
  

*Economic situation
gen pers_econ_worse = finance_finpast
recode pers_econ_worse (-9/-8=.) (3=2) (2=3)
  
*Distrust in government
gen distrust_gov = trustgov_trustgrev
recode distrust_gov (-9/-1=.) 
 
*Age*
gen age = dem_age_r_x
recode age (-2=.) 
 
*Gender*
gen female = gender_respondent_x
recode female (2=1) (1=0) 
 
*Education*
gen edu = dem_edugroup_x
recode edu (-9/-2=.)
 
*Ethnicity*
gen white_nonhispanic = dem_raceeth_x
recode white_nonhispanic (-9=.) (2/6=0)

*state*
gen state = sample_stfips

**variables' normalisation (0 to 1)**
 
gen lib_cons_norm = (lib_cons - 1) / 6
gen Disapp_economy_norm = Disapp_economy - 1
gen pers_econ_worse_norm = (pers_econ_worse - 1) / 2
gen Distrust_gov_norm = (distrust_gov  - 1)/4
gen age_norm = (age - 18) / (90-18)
gen edu_norm = (edu - 1) / 4


*logit model - vote intention (Table A1)*

logit vote_Romney lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state , vce(robust)
logit, or

predict yhat_pre if vote_Romney !=. & lib_cons_norm !=. & Disapp_economy_norm !=. & pers_econ_worse_norm !=. & Distrust_gov_norm !=. & age_norm !=. & edu_norm !=. & female !=. & white_nonhispanic !=. & state !=.

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
gen predicted_Romney_pre_vote2 = 0
recode predicted_Romney_pre_vote2 (0=1) if yhat_pre > .5
replace predicted_Romney_pre_vote2 = . if yhat_pre == . 


***b) replication with average predicted probability
svyset [pweight=weight_full]
svy: mean yhat_pre
gen predicted_Romney_pre_vote3 = 0
recode predicted_Romney_pre_vote3 (0=1) if yhat_pre > .4429092
replace predicted_Romney_pre_vote3 = . if yhat_pre == . 



*replication with Bayesian logistic model (Table A2)*


bayesmh vote_Romney lib_cons_norm Disapp_economy_norm Distrust_gov_norm age_norm edu_norm i.white_nonhispanic pers_econ_worse_norm i.female i.state , ///
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
    mcmcsize(1000) burnin(5000) rseed(12345) saving(mcmc2012, replace)
	
	bayespredict vote_Romney_p , mean rseed(12345)

***a) generating predicted popular vote shares for Trump with 0.5 predicted probability threshold
	gen simul_Romney_vote = 0
	recode simul_Romney_vote (0=1) if vote_Romney_p >  .5
	replace simul_Romney_vote = . if vote_Romney_p == . 

***b) replication with average predicted probability
	svy: mean vote_Romney_p
	gen simul_Romney_vote2 = 0
	recode simul_Romney_vote2 (0=1) if vote_Romney_p  >  .4478082
	replace simul_Romney_vote2 = . if vote_Romney_p  == . 

***c) Aggregation and comparison of predicted votes according to various strategies (Table 1)
mean vote_Romney
svy: mean vote_Romney
svy: mean predicted_Romney_pre_vote2  
svy: mean predicted_Romney_pre_vote3
svy: mean simul_Romney_vote 
svy: mean simul_Romney_vote2
