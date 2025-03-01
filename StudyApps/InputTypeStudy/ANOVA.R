library(reshape2)
library(readr)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(plyr)
library(datarium)
library(afex)
library(ggplot2)
library(emmeans)
library(ggbeeswarm)
library(multcomp)
library(ez)
library(psych)
library(nlme)
data("selfesteem2", package = "datarium")
selfesteem2 <- selfesteem2 %>%
  gather(key = "time", value = "score", t1, t2, t3) %>%
  convert_as_factor(id, time)
res.aov <- anova_test(
  data = selfesteem2, dv = score, wid = id,
  within = c(treatment, time)
)
get_anova_table(res.aov)

library(apaTables)

data <- read_csv("Documents/GitHub/HeadEyeTracking/StudyApps/InputTypeStudy/newstudy_BySubject.csv")
dwell_data = read_csv(file='DwellRawSummary.csv')
dwell_data <- within(dwell_data,{
  dwell_data$subject_num <-factor(dwell_data$subject_num)
  dwell_data$posture <-factor(dwell_data$posture)
  dwell_data$cursor_type <- factor(dwell_data$cursor_type)
  dwell_data$dwell_time <- factor(dwell_data$dwell_time)
})
data <- within(data,{
  data$subject <-factor(data$subject)
  data$posture <-factor(data$posture)
  data$cursor <- factor(data$cursor)
})
library(rlang)

print_basic_statistics <- function(col){
  print(col)
  data <- read_csv("Documents/GitHub/HeadEyeTracking/StudyApps/InputTypeStudy/newstudy_BySubject.csv")
  #data = data[-which(is.na(data$target_in_count)), ]
  data <- within(data,{
    data$subject <-factor(data$subject)
    data$posture <-factor(data$posture)
    data$cursor <- factor(data$cursor)
    data$selection <- factor(data$selection)
  })
  data.mean <- aggregate(data[col],
                         by=list(data$subject,data$posture,data$cursor,data$selection),
                         FUN='mean', na.rm=TRUE)
  colnames(data.mean)<- c("subject","posture","cursor","selection","x")
  data.aov<-ezANOVA(data=data.mean, dv=x, wid=subject, within=c(posture, cursor,selection))
  print(apa.ezANOVA.table(data.aov))
  # colnames(data.mean)<- c("subject_num","posture","cursor_type","x")
  a1 <- aov_ez(id="subject",dv= "x", data=data.mean, within=c("posture", "cursor","selection"), anova_table = list(es = "pes"))
  # switched to pes (partial eta squared) for compatability with ART
  print(a1)
  summary(a1)$sphericity.tests
  apa_print(a1, correction = "none", es="pes")$full_result 													   # get latex and effect sizes with NO spherecity corrections
  apa_print(a1, correction = "GG", es="pes")$full_result 	
  data.aov <- with(data.mean,
                   aov(x ~ posture * cursor_type +
                         Error(subject / (posture * cursor)))
  )
}
print_basic_statistics('success')

print_dwell_statistics <- function(col,stand=TRUE,print_result=TRUE,print_posthoc=TRUE){
  print(col)
  dwell_data = read_csv(file='DwellRawSummary.csv',show_col_types = FALSE)
  data = data[-which(is.na(data$target_in_count)), ]
  
  if (stand){
    dwell_data <-dwell_data[dwell_data$posture=='STAND',]
  }else{
    dwell_data <-dwell_data[dwell_data$posture=='WALK',]
  }
  dwell_data <- within(dwell_data,{
    dwell_data$subject_num <-factor(dwell_data$subject_num)
    dwell_data$posture <-factor(dwell_data$posture)
    dwell_data$cursor_type <- factor(dwell_data$cursor_type)
    # dwell_data$dwell_time <- factor(dwell_data$dwell_time )
   dwell_data$dwell_time <- factor(dwell_data$dwell_time ,levels=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0), labels=c("0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"))
  })
  
  dwell_data.mean <-aggregate(dwell_data[col],
                              by=list(dwell_data$cursor_type,dwell_data$dwell_time),
                              FUN='mean', na.rm=TRUE)
  colnames(dwell_data.mean)<- c("subject_num","cursor_type","dwell_time","x")
  dwell_data.mean$dwell_time <- factor(dwell_data.mean$dwell_time ,levels=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0), labels=c("0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"))
  dwell_data.aov<-ezANOVA(data=dwell_data.mean, dv=x, wid=subject_num, within=c( cursor_type,dwell_time), white.adjust = TRUE
                    , detailed = FALSE
                    , return_aov = FALSE )
  
  a1 <- aov_ez(id="subject_num",dv= "x", data=dwell_data.mean, within=c("cursor_type","dwell_time")
               , anova_table = list(es = "pes")
               )
  if(print_result){
  print(dwell_data.aov)
  print('A1')
  print(a1)
  # print("APAPRINT")
  # print(apa_print(a1, correction = "none", es="pes")$full_result) 													   # get latex and effect sizes with NO spherecity corrections
  # print(apa_print(a1, correction = "GG", es="pes")$full_result) 														   # get latex and effect sizes WITH spherecity corrections
  }
  if(print_posthoc){
  # m1=emmeans(a1,~ dwell_time)
  # print(summary(as.glht(pairs(m1)), test=adjusted("bonferroni")))
  # m1=emmeans(a1,~ cursor_type)
  # print(summary(as.glht(pairs(m1)), test=adjusted("bonferroni")))
    # pwc <-dwell_data.mean %>% group_by(cursor_type) %>%
    #   emmeans_test(x ~ dwell_time, p.adjust.method = 'bonferroni')
    # print(pwc,n=135)
    m1=emmeans(a1,~ cursor_type)
    print(summary(as.glht(pairs(m1)), test=adjusted("bonferroni")))
    print(dwell_data.mean %>%
            group_by(cursor_type) %>%
            get_summary_stats(x,type='mean_sd')
    )
  }
}
for (cn in c(
  'success_trial', 'required_target_size', 'trial_time',
  'final_speed', 'target_in_count'
     ))
  {
  print_dwell_statistics(cn,stand=TRUE,print_result=TRUE,print_posthoc = TRUE)
}
for (cn in c(
  'success_trial', 'required_target_size'
  # , 'trial_time'
  # ,'final_speed', 'target_in_count'
))
{
  print_dwell_statistics(cn,stand = FALSE,print_result=TRUE,print_posthoc = TRUE)
}  
  

data <-data[order(data$subject_num),]
data = data[-which(is.na(data$target_in_count)), ]
data.mean <- aggregate(data$success_trial,
                       by=list(data$subject_num,data$posture,data$cursor_type),
                       FUN='mean')
colnames(data.mean) <- c("subject_num","posture","cursor_type","success_trial")
ezANOVA(data=data.mean, dv=success_trial, wid=subject_num, within=c(posture, cursor_type) )





dwell_data.mean <-aggregate(dwell_data$success_trial,
                            by=list(dwell_data$subject_num,dwell_data.dwell_time,dwell_data$cursor_type),
                            FUN='mean')
library(afex)
Model.1<-aov_car(success_trial ~ dwell_time * cursor_type +
                   Error(subject_num / (dwell_time*cursor_type)), 
                 data = dwell_data.mean,include_aov = TRUE)
Model.1


c = 'success_trial'
data.mean <- aggregate(data$success_trial,
                       by=list(data$subject_num,data$posture,data$cursor_type),
                               FUN='mean')
colnames(data.mean) <- c("subject_num","posture","cursor_type","success_trial")


data$posture = factor(data$posture,levels=unique((data$posture)))
data$cursor_type = factor(data$cursor_type,levels=unique((data$cursor_type)))
model = lme(success_trial ~ posture + cursor_type + posture*cursor_type,data=data)

ezANOVA(data=data.mean, dv=success_trial, wid=subject_num, within=c(posture, cursor_type) )
data$posture = as.factor(data$posture)
data$cursor_type = as.factor(data$cursor_type)
data$wide = as.factor(data$wide)


row.names(data) = NULL


data.mean <- aggregate(data,
                         by = list(data$subject_num, data$posture,
                                   data$cursor_type),
                         FUN = 'mean'
                       ,na.rm=TRUE
                       )
colnames(data.mean) = c('subject_num','posture','cursor_type',c)
colnames(data.mean) = c('subject_num','posture','cursor_type','wide',c)
data.aov <- with(data.mean,
                   aov(initial_contact_time ~ posture * cursor_type +
                         Error(subject_num / (posture * cursor_type)))
)
summary(data.aov)
a1 = aov_ez('subject_num','mean_offset',data.mean,within=c('posture','cursor_type'),
            anova_table = list(es = "pes")
            # , fun_aggregate = mean,na.rm=TRUE
            )
print(a1)
summary(a1)$sphericity.tests
library("papaja")
apa_print(a1, correction = "none", es="pes")$full_result 													   # get latex and effect sizes with NO spherecity corrections
apa_print(a1, correction = "GG", es="pes")$full_result
# m1 = emmeans(a1,~posture)
afex_plot(a1, x = "posture",error="within")																		
afex_plot(a1, x = "cursor_type",error="within")
m1=emmeans(a1,~ cursor_type)
summary(as.glht(pairs(m1)), test=adjusted("bonferroni"))
as.glht(pairs(m1))
print("****** TAP ******")
twoWayRM = featureData[featureData$action=="Tap"&featureData$type=="Single",c("subject","duration","nail","nailDetail")]
twoWayRM <-aggregate(twoWayRM$duration, by=list(twoWayRM$subject,twoWayRM$nailDetail,twoWayRM$nail), FUN=mean, na.rm=TRUE)
colnames(twoWayRM) = c("subject","nailDetail","nail","duration")
a1 <- aov_ez("subject", "duration", twoWayRM, within = c("nail","nailDetail"), anova_table = list(es = "pes")) # switched to pes (partial eta squared) for compatability with ART
summary(a1)$sphericity.tests 																				   # check mauchley - if <0.05, apply corrections
apa_print(a1, correction = "none", es="pes")$full_result 													   # get latex and effect sizes with NO spherecity corrections
apa_print(a1, correction = "GG", es="pes")$full_result 														   # get latex and effect sizes WITH spherecity corrections
m1 <- emmeans(a1, ~ nail) 																			  		   # get pairs for nail
summary(as.glht(pairs(m1)), test=adjusted("bonferroni")) 													   # apply corrections and report sig
m1 <- emmeans(a1, ~ nailDetail) 																			   # get pairs for nail detail
summary(as.glht(pairs(m1)), test=adjusted("bonferroni")) 													   # apply corrections and report sig
#plot(twoWayRM$nailDetail, twoWayRM$duration)							
afex_plot(a1, x = "nail",error="within")																		
afex_plot(a1, x = "nailDetail",error="within")




data <- read.csv("BasicRawSummary.csv")
# data$posture = as.factor(data$posture)
# data$cursor_type = as.factor(data$cursor_type)

data = data[-which(is.na(data$target_in_count)), ]
row.names(data) = NULL

cols = colnames(data)[13:25]
out = data.frame()
for (var in cols) {
  var_name = paste0("mean_", var)
  data.ddply = ddply(data, .(subject_num, posture, cursor_type, repetition) , summarize, mean_var = mean(eval(parse(text = var))))  
  
  data.ddply <- data.ddply %>% mutate(id = paste0(subject_num,"_", repetition))
  result <- data.ddply %>% anova_test(mean_var ~
                                        posture * cursor_type +
                                        Error(id/(posture*cursor_type)))
  
  sphericities = result[["Mauchly's Test for Sphericity"]][["p"]]
  pVals = c(paste0(result[["ANOVA"]][["DFn"]][1], ", ",result[["ANOVA"]][["DFd"]][1]), result[["ANOVA"]][["p"]][1])
  for (i in 1:length(sphericities)) {
    if (sphericities[i] < 0.05) {
      pVals = c(pVals, result[["Sphericity Corrections"]][["DF[GG]"]][i], result[["Sphericity Corrections"]][["p[GG]"]][i], "TRUE")
    } else {
      pVals = c(pVals, paste0(result[["ANOVA"]][["DFn"]][i+1], ", " , result[["ANOVA"]][["DFd"]][i+1]), result[["ANOVA"]][["p"]][i+1], "FALSE")
    }
  }
  
  out = rbind(out, c(var_name, pVals))
}

colnames(out) = c("var","df_posture", "p_posture","df_cursor",  "p_cursor", "sphericity_correction_cursor","df_posture*cursor", "p_posture*cursor", "sphericity_correction_posture*cursor")