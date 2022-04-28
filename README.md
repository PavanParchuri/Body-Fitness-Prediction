# Body-Fitness-Prediction

### INTRODUCTION: 
Sedentary lifestyle is defined by the absence of physical activity practices throughout the day and causes a decrease in caloric expenditure. This behaviour is explained by the inappropriate lifestyle, for example, too much time sitting or lying down and still eating unhealthy foods during this time of immobilization. Currently, a third of the adult world population is physically inactive and this generates five million deaths per year (The Lancet, 2012). Additionally contributing to several chronic diseases, physical inactivity also influences mood, sleep quality and body weight.


Our system focuses more on user friendliness of all type of people and they can access anywhere. A user can give inputs such as step count, mood, calories burned, hours of sleep and weight in the website. We take those values and give it to machine learning model. Finally, it will predict whether the person is active or inactive based on their given data.


### OBSERVATIONS: 
●	It was observed that when comparing the level of physical activity by the step count, people who are "happy" showed greater level when compared to the categories of "sad" and "neutral".

●	In addition, "happy" people spend more energy (kilo calories) than "neutral" or "sad" people.

●	"Happy" people demonstrate sleeping more hours of sleep when compared to the other categories. However, "sad" people also sleep more hours than "neutral" people.

●	Self-perceived activity (active / inactive) also demonstrates differences in the case of the level of physical activity (by counting steps), "active" people tend to walk more steps, spend more calories and sleep more hours when compared to people self - called "inactive".

●	From the given data there is no much variance in weights. However, some how we can say that less weight people tends to be more "happy" which in turn effects on activeness.

●	Finally, the association between self-perception of activity with Mood, 46 percent of the people considered "sad" are "inactive" and 57 percent of the "happy" people are considered "active.

Jessica Selinger's study reports on the "law of least effort" and demonstrates that the body adjusts to the least effort, as the brain boycotts efforts to save energy costs. Therefore, physically demanding of the body improves health and increases the disposition. It is known that the practice of regular physical activity is a leading intervention for better physical and mental health.


### PROJECT STRUCTURE:
This project has four parts:
1. model.py — This contains code for the machine learning model to predict whether a
person is active or inactive based on hours of sleep, body weight, mood (sad, neutral,
and happy), steps count, and caloric expenditure.
2. app.py — This contains Flask APIs that receives sales details through GUI or API
calls, computes the predicted value based on our model and returns it.
3. request.py — This uses requests module to call APIs defined in app.py and displays
the returned value.
4. HTML/CSS — This contains the HTML template and CSS styling to allow user to
enter inputs and displays the physical activeness of the user (active/inactive).


### CONCLUSION: 
In this model, we are detecting whether the person is active or inactive based on steps count,
mood, body weight and hours of sleep using Gradient Boosting Classifier model. The purpose of this
application is to view the existing technology of machine learning in for health care and use present
technology for the development by which the user can check his health. Our project will create a
better environment and it can be used very effectively for better body fitness and health. It is
useful for people who takes their health seriously and wants to be fit as they can check daily
and review their activeness. With this application, a person can easily know whether he/she is
active or not, can check his body fitness regularly to maintain better health, can improve person
lifestyle so that he can recover a bit than before, easy to access this application, a person can
maintain proper fitness with the help of this application.
