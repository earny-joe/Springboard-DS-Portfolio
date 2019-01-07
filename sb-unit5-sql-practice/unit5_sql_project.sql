/*
The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */

/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT `name`,`membercost`
FROM `Facilities` 
WHERE `membercost` != 0

Output (name, membercost):
- Tennis Court 1, 5.0
- Tennis Court 2, 5.0
- Massage Room 1, 9.9
- Massage Room 2, 9.9
- Squash Court, 3.5

/* Q2: How many facilities do not charge a fee to members? */
SELECT `name`,`membercost`
FROM `Facilities` 
WHERE `membercost` = 0

Output (name, membercost):
- Badminton Court, 0.0
- Table Tennis, 0.0
- Snooker Table, 0.0
- Pool Table, 0.0

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT `facid`,`name`,`membercost`,`monthlymaintenance`
FROM `Facilities` 
WHERE `membercost` < 0.20*`monthlymaintenance`

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */
SELECT `facid`, `name`, `membercost`, `guestcost`, `initialoutlay`, `monthlymaintenance`
FROM `Facilities` 
WHERE `facid` IN (1, 5)

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */
SELECT `name`,  
       `monthlymaintenance`,
       CASE WHEN `monthlymaintenance` > 100 THEN 'expensive'
       ELSE 'cheap' END AS cheap_or_exp
  FROM `Facilities` 
 ORDER BY `monthlymaintenance` DESC

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */
SELECT `surname`, `firstname`, `joindate`
FROM `Members`
ORDER BY `joindate` DESC

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT CONCAT(m.firstname, ' ', m.surname) AS member_name, f.name AS facility_name
FROM country_club.Members m
INNER JOIN country_club.Bookings b ON m.memid = b.memid
INNER JOIN country_club.Facilities f ON b.facid = f.facid
WHERE f.facid IN ('Tennis Court 1', 'Tennis Court 2')
GROUP BY member_name

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */
SELECT members.surname AS member_name, facilities.name AS facility_name, facilities.guestcost * bookings.slots AS booking_cost
FROM country_club.Bookings bookings
JOIN country_club.Facilities facilities ON bookings.facid = facilities.facid
JOIN country_club.Members members ON members.memid = bookings.memid
WHERE LEFT(bookings.starttime, 10) = '2012-09-14'
AND members.memid = 0
UNION
SELECT CONCAT(members.firstname, ' ', members.surname) AS member_name, facilities.name AS facility_name, SUM(facilities.membercost * bookings.slots) AS booking_cost
FROM country_club.Bookings bookings
JOIN country_club.Facilities facilities ON bookings.facid = facilities.facid
JOIN country_club.Members members ON members.memid = bookings.memid
WHERE LEFT(bookings.starttime, 10) = '2012-09-14'
AND members.memid != 0
HAVING booking_cost > 30
ORDER BY booking_cost DESC

/* Q9: This time, produce the same result as in Q8, but using a subquery. */


/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */
