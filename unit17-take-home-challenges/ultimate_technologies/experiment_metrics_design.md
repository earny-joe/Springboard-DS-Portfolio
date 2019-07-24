# Part 2 - Experiment and Metrics Design

The neighboring cities of Gotham and Metropolis have complementary circadian rhythms: on weekdays, Ultimate Gotham is most active at night, and Ultimate Metropolis is most active during the day. On weekends, there is reasonable activity in both cities.

However, a toll bridge, with a two-way toll, between the two cities causes driver partners to tend to be exclusive to each city. The Ultimate managers of city operations for the two cities have proposed an experiment to encourage driver partners to be available in both cities, by reimbursing all toll costs.

1. What would you choose as the key measure of success of this experiment in encouraging driver partners to serve both cities, and why would you choose this metric?
2. Describe a practical experiment you would design to compare the effectiveness of the proposed change in relation to the key measure of success. Please provide details on:
    a. how you will implement the experiment
    b. what statistical test(s) you will conduct to verify the significance of the observation
    c. how you would interpret the results and provide recommendations to the city operations team along with any caveats.
    

## Response:

My first response would be to question why the managers want to encourage driver-partners to be available in both cities. This is not to say that it isn't valid, but we need to assess the value it's going to bring to not only the company but to the drivers as well, as these drivers are going to incur higher costs (i.e., gas, time spent traveling and not working). 

If we're in a situation where we don't have the supply of drivers necessary to meet the demand for rides than trying to encourage more drivers from the other city might prove a useful strategy. However, as a preliminary experiment, we should test to see if our supply (of drivers) is meeting the demand (of rides). A way to check this might be to utilize total wait times, as a lower wait time usually indicates that there is enough supply of drivers and the only time cost is that associated with the driver driving to their location. A one-tailed z-test is necessary here (as you can't have a waiting time less than 0 minutes) because we can easily obtain a sample size greater than 30 observations. Additionally, information related to waiting time (including mean and standard deviation) is most likely available in Ultimate's database. In this case, our null hypothesis would be that there is no difference in wait time between the two cities. However, if this turns out to not be the case - i.e., we reject the null hypothesis and favor the alternative, which is that there is a difference in wait times - then we can explore subsidizing the toll cost. 

Why I think it is essential to take a step back in a sense before offering the reimbursement is that in both situations - subsidizing the toll without testing supply/demand & testing supply/demand to discover a lack of driver supply - there is a cost associated with it. However, the costs are different:

- The cost associated with the lack of driver supply indicates a situation of lost revenue (since we are losing out on potential revenue from riders that would otherwise use the service if they could get a ride within an acceptable wait time).
- The cost associated with the reimbursement is directly relayed in higher business costs since these tolls need to be paid out to the drivers. However, if there is no demand for the extra drivers (i.e., we're not losing out on potential revenue from riders) we are effectively raising our operational costs without any benefit. 

In my opinion, it is better to pursue option 1 as the cost is not directly related to operational costs as the margin between revenue and expenses will stay the same as opposed to option 2, which will cause operational costs to increase due to the toll reimbursement without a subsequent increase in revenue (since the extra drivers are idle in the other city). 

If we do encounter the situation where the supply of drivers is not meeting the demand, we could then explore experimenting with toll reimbursement. I still think the wait time is the most appropriate measure here. If we were not able to reject our initial hypothesis after implementing the experiment (i.e., cannot reject the null hypothesis that wait time is different in either city), then this indicates we're meeting the demand within each municipality. The last thing I have to say though is that then we would need to deem if the increased revenue is enough to offset the increased cost associated with the toll reimbursement. 




