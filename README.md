# Job Descriptions Analysis
## Overview
 
This project intend to analyze latent features (topics) in job descriptions. It answers questions such as how valuable a skill-set is, how much the value varies across companies and What topics or skills contribute to high salaries. Average salaries per topic has been used to determine the value of topics or skills. Jobs are ranked for each topic allowing to determine the relevance strength of skills for each topic. 

  
## Data

The data has been collected from different sources and merged for matching job titles.

### Data sources:

*  Salaries [Salary, Date, State, City]:  [www.jobs-salary.com](http://www.jobs-salary.com)
*  Job descriptions: [indeed.com](http://www.indeed.com) and [simplyhired.com](http://www.simplyhired.com)

The Dataset consist of 15724 jobs for 12 companies and 88334 salaries.

![image](https://cloud.githubusercontent.com/assets/13112596/10216841/c937e094-67e0-11e5-8077-597d968062f2.png)
<img src=https://cloud.githubusercontent.com/assets/13112596/10216871/0e8e8bde-67e1-11e5-9c87-3a39deba4aa1.png width=400 height=400 />
<img src=https://cloud.githubusercontent.com/assets/13112596/10216872/149890f6-67e1-11e5-919b-02f30d83805f.png width=400 height=400 />
## Latent Feature example
The generated matrixes from NMF rank jobs per latent feature and words per the same latent feature. This makes each latent feature identifiable by the words it ranks highly. The following are examples of some latent features: 

### Customer services:
The following image, illustrates the top 400 words and their sizes represent their word rank for the this feature:
![image](https://cloud.githubusercontent.com/assets/13112596/10216879/1ad23c2e-67e1-11e5-9f8e-f0dccf0642a0.png)

![image](https://cloud.githubusercontent.com/assets/13112596/10217515/2b0942e6-67e5-11e5-8fb5-6d575d73ebc1.png)

The X-axes represent the strength/weight of this latent feature (Customer services) per job, while y-axis is the corresponding average salary. It shows that the more the job description is relevant to customer services the lower the average salaries.

