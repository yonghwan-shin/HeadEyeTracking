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
data("selfesteem2", package = "datarium")
selfesteem2 <- selfesteem2 %>%
  gather(key = "time", value = "score", t1, t2, t3) %>%
  convert_as_factor(id, time)
res.aov <- anova_test(
  data = selfesteem2, dv = score, wid = id,
  within = c(treatment, time)
)
get_anova_table(res.aov)


data <- read_csv(file="BasicRawSummary.csv")
data$posture = as.factor(data$posture)
data$cursor_type = as.factor(data$cursor_type)

data = data[-which(is.na(data$target_in_count)), ]
row.names(data) = NULL


data.mean <- aggregate(data$initial_contact_time,
                         by = list(data$subject_num, data$posture,
                                   data$cursor_type),
                         FUN = 'mean'
                       # ,na.rm=TRUE
                       )
colnames(data.mean) = c('subject_num','posture','cursor_type','initial_contact_time')
data.aov <- with(data.mean,
                   aov(initial_contact_time ~ posture * cursor_type +
                         Error(subject_num / (posture * cursor_type)))
)
summary(data.aov)
a1 = aov_ez('subject_num','initial_contact_time',data.mean,within=c('posture','cursor_type'), 
            # anova_table = list(es = "pes")
            )
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