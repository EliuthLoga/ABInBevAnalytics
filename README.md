# ABInBevAnalytics

This repository contains a solution to the problem of generate recommendations to users 
into the AB InBev Analytics  application.

### Running

Use pipreqs to automatically generate a requirements.txt file based on the import statements that the Python script(s) contain. 

```
pip install pipreqs
pipreqs .
```

Commands above will generate a requirements.txt, containing information for libraries to be installed. 

Finally, install using:

```
pip install -r requirements.txt
```

Run the recommender system application

```
com/analitics/application/RecommendationApp.py
```
