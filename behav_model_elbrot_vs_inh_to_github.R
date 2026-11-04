library(mgcv)
library(nlme)
library(abind)
library(gratia)
library(ggeffects)
library(emmeans)
library(tidyverse)
library(itsadug)
library(dplyr)
library(pals)

subjects <- paste0('BM_C', sprintf('%02d', setdiff(3:27, c(13))))

df_subj <- data.frame()
for (subj in subjects) {
  print(subj)
  df <- read.csv(paste0('/for_r/', subj, '_SMA-med_test_zscored_man_rot.csv'))
  df_subj <- rbind(df_subj, df)
}
df_subj$Subject <- as.factor(df_subj$Subject)

df_subj_filtered <- df_subj %>%
  filter(Inh_time>-0.07, Inh_time <= 0.15)

###########################################################

#### FINAL MODEL
model_inh3 <- bam(PeakElbRot ~ s(Inh) + s(Inh_time) + s(Inh_onset) + s(Inh_dur) + 
                    ti(Inh_dur,Inh_onset) + ti(Inh,Inh_onset) + ti(Inh,Inh_dur) + 
                    ti(Inh_dur,Inh_time) + ti(Inh,Inh_time) + ti(Inh_onset,Inh_time) +
                    s(Subject, Inh, bs='re') +  s(Subject, Inh_dur, bs='re') + s(Subject, Inh_onset, bs='re') + s(Subject, Inh_time, bs='re') + 
                    s(Subject, bs='re'), 
                  data = df_subj_filtered,
                  method = 'fREML')
summary(model_inh3)
plot_smooth(model_inh3, view = "Inh_time", rug = TRUE)
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]", "Inh_dur [-17.7:-0.62, by=1]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh peak-time, s') + ylab('Peak ElbRot')  +labs(color = "Inh dur")
(ggemmeans(model_inh3, terms = c("Inh_onset [-0.1:0.1, by=0.025]", "Inh_dur [-17.7:-0.62, by=1]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh onset-time, s') + ylab('Peak ElbRot')  +labs(color = "Inh dur")
(ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.025]", "Inh [0:3, by=0.25]"))) %>% ggplot(aes(x = x, y = predicted, ymin=conf.low, ymax=conf.high, color=as.numeric(as.character(group)), group = group)) +     geom_line() + scale_color_viridis_c()+ xlab('Inh peak-time, s') + ylab('Peak ElbRot')  +labs(color = "Inh Ampl")

# Generate predicted values for the smooth term
pred <- ggemmeans(model_inh3, terms = "Inh_time")

# Plot
p <- (ggplot(pred, aes(x = x, y = predicted)) +
  geom_line(color = "black", size = 1) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  geom_rug(data = df_subj_filtered, aes(x = Inh_time), inherit.aes = FALSE) +
  labs(x = "Inhibition Peak Time, s",
       y = "Predicted Peak Elbow Rotation") +
  theme_minimal(base_size = 16))

plot(p)  # built-in method draws ggplot
ggsave("elbrot_vs_inh_behav_inhtime_maineffect.svg", plot = p, width = 6, height = 4, units = "in")


# 1. Draw from the posterior of the smooth over grid

draws <- smooth_samples(
  model_inh3,
  select = "s(Inh_time)",
  n = 1000,
  newdata = data.frame(
    Inh_time = seq(
      min(df_subj_filtered$Inh_time),
      max(df_subj_filtered$Inh_time),
      length.out = 500
    )
  ),
  seed = 42
)

# 2. Find the argmin (peak of U shape = minimum of smooth) per draw
argmin_per_draw <- draws |>
  group_by(.draw) |>
  slice_min(order_by = .value, n = 1, with_ties = FALSE) |>
  pull(Inh_time)

# 3. Summarise in median and 95% CI

q <- quantile(argmin_per_draw, probs = c(0.025, 0.5, 0.975))
df_subj_filtered$Inh_time[round(q)]

#be careful: transformed values changed polarity

#MAKE A PLOT

#for paper
full_colors <- parula(100)

# keep only the last ~50% of the colors (orange → red)
subset_colors <- full_colors[0:90]

# Now use in ggplot
p <- (ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.015]",
                                      "Inh [-0.06:3.5, by=0.15]"))) %>%
  ggplot(aes(x = x,
             y = predicted,  # use -predicted if you want stronger inhibition negative
             color = as.numeric(as.character(group)),
             group = group)) +
  geom_line(size = 1.2) +
  
  # Color scale
  scale_color_gradientn(
    colors = subset_colors,
    name = "Peak Inhibition"
  ) +
  
  # Labels
  xlab("Inhibition peak, s (0-unloading)") +
  ylab("Peak Elbow Rotation") +
  scale_y_continuous(limits = c(0.0045, 0.0125)) +
  
  # Matplotlib-style theme
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),  # subtle grid
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )
plot(p)  # built-in method draws ggplot

# Get parula colors
full_colors <- parula(100)
subset_colors <- full_colors[5:95]

# Generate plot for Inh_time (x) and Inh (color)
p <- ggemmeans(model_inh3, terms = c("Inh_time [-0.07:0.15, by=0.015]",
                                     "Inh [-0.68:3.5, by=0.25]")) %>%
  ggplot(aes(
    x = x,
    y = predicted,
    color = as.numeric(as.character(group)),
    group = group
  )) +
  geom_line(size = 1.2) +
  scale_color_gradientn(colors = subset_colors, name = "Peak Inhibition") +
  xlab("Inhibition peak-time, s") +
  ylab("Peak Elbow Rotation") +
  theme_bw(base_size = 19, base_family = "DejaVu Sans") +
  theme(
    panel.grid.major = element_line(color = "grey90", size = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    axis.ticks = element_line(color = "black"),
    axis.text = element_text(color = "black", size = 15),
    axis.title = element_text(size = 19),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 15)
  )

plot(p)
ggsave("elbrot_vs_inh_behav_inh_vs_inhtime.svg", plot = p, width = 8, height = 4, units = "in")