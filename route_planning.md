# Route Planning Task
### Report

##### Our topic is the Waiter Project. To complete the Route planning Task we have decided to use the most intuitive way to traversal all the tables available on the grid. Beacuse we already have a 16x16 grid, we do not think about extra search strategies which are for trees or graphs.

##### First, we put all the point of tables in an array. For example: [table1, table2, ......].
##### In a while loop, it starts at picking the first element(point) in the array and calculating the distance between the agent and the point, and choose the smallest distance of four directions(left, right, up, down). Then move to that direction.
##### If the agent arrive the destination(means the agent and the table are at the same point), then the table in array will be deleted. And the agent will go back to kitchen.
##### When agent arrive the ketchen, the loop will iterate again to the next table until all the tables are visited.
