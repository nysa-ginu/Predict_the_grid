# Predict_the_grid

<h3>Introduction</h3>

<p>Formula-1 has been gaining a lot of traction in the recent years, especially in the US. Being the pinnacle of motorsports, it represents the epitome of speed, technology and skills. The sport pushes the boundaries of engineering and innovation, with teams constantly striving to design and develop the fastest and most technologically advanced cars on the planet. The drivers, known for their exceptional skill and courage, showcase their talent by maneuvering these powerful machines at incredible speeds, navigating challenging tracks and battling fierce competition. Formula-1's global appeal, glamorous race weekends, and unparalleled level of competition have solidified its status as the ultimate stage for motorsports, captivating fans and inspiring generations of racing enthusiasts.</p>
<p>From a more data-centric viewpoint, this sport creates a vast amount of data which is in turn used by the teams and drivers to plan their races. Be it developing their cars or optimize their strategies, including car setup, pit-stop timing, and race tactics, data has played a crucial role in this sport.
<br>
Creating a Formula-1 races prediction model enables us to harness the power of data and statistical modeling to gain insights, enhance decision-making, and fuel the enthusiasm and competitiveness of this thrilling motorsport.</p>

<h3>Data Collection</h3>

<p>The data used for this project was taken from <a href="https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=qualifying.csv">Kaggle</a>. Apart from historical race data, the data on car performance and weather also play a crucial role in predicting the outcome of a race. Since the car performance data is not shared by the teams, it was crucial to get the weather data. The weather data was obtained using the OpenWeatherMap API.</p>

<h3>Method</h3>

<p>While selecting a model, it was important to classify if this is a regression or classification problem. Since I only wanted to predict the drivers who will be in the podium psitions, i.e., 1st, 2nd and 3rd, I chose to convert this into a multi-class classification problem and planned to use Logistic Regression, Random Forest and XGBoost models.</p>

<h3>Conclusion</h3>

<p>After comparing the results of each of the model, Random Forest performed the best with a score of 0.843182.</p>
