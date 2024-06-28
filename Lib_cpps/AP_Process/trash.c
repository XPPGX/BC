#pragma region DETECT_ARTICULATION_POINT

void apInitSequential(Articulation_point_sequential* articulation_point, GraphInfo* graph_info)
{
    // Initial AP record data strucutre

        // Malloc recording array
    articulation_point -> discovery_time = (int*)malloc((graph_info -> num_node) * sizeof(int));
    articulation_point -> low_time = (int*)malloc((graph_info -> num_node) * sizeof(int));
    articulation_point -> parent = (int*)malloc((graph_info -> num_node) * sizeof(int));
    articulation_point -> ap_flag = (bool*)malloc((graph_info -> num_node) * sizeof(bool));
    articulation_point -> bicomponent_id = (int*)malloc((graph_info -> num_node) * sizeof(int));

    #ifdef ERROR_TEST
    if(!(articulation_point -> discovery_time) || !(articulation_point -> low_time) || !(articulation_point -> parent) || !(articulation_point -> ap_flag) || !(articulation_point -> bicomponent_id))
    {
        writeErrorLog("Error occured : Malloc in apInit() failed.");
        cerr << "CHECK error.log !!!" << endl;
    }
    #endif /* ERROR_TEST */


        // Initialization
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        articulation_point -> discovery_time[node_id] = 0;
        articulation_point -> low_time[node_id] = 0;
        articulation_point -> parent[node_id] = -1;
        articulation_point -> ap_flag[node_id] = 0;
        articulation_point -> bicomponent_id[node_id] = -1;
    } 
    
    articulation_point -> num_ap = 0;
    articulation_point -> num_commponent = 0;
}

void apSeqDetect(GraphInfo* graph_info, Graph* graph, Articulation_point_sequential* articulation_point)
{

    // Initial ap_stack
    STACK* ap_stack = stackInitial((int)((graph_info -> num_node) * 0.5));
    // cout << "Initial stack size = " << (int)((graph_info -> num_node) * 0.5) << endl;

    // Start to do DFS traversing to find AP
        // Initial using variable
    int time = 0;                    // For recording the time of the finding AP
    bool backward_trace_flag = true; // For  recording whether DFS can keep doing or not, "false" means there is the end of the DFS path, otherwise, there exist neighbor never been visited

        // Consider whether dataset exist more than 1 component(dataset is disconnected)
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        // If it has been visited, then just pass doing DFS
        if(articulation_point -> discovery_time[node_id]) continue;

        // Initial visit time
        time = 0;

        // Push the starting node into stack
        stackPush(ap_stack, node_id);
        articulation_point -> discovery_time[node_id] = articulation_point -> low_time[node_id] = ++time;

        // DFS
        while(!stackEmpty(ap_stack))
        {

            int recent_node = StackGetTop(ap_stack);    // Don't pop, cause we will do backward tracing

            // Assume exist neighbor hasn't been visited
            backward_trace_flag = true;

            // Start to consider neighbors of recent_node
            for(int edge_id = graph -> offset[recent_node] ; edge_id < graph -> offset[recent_node + 1] ; edge_id++)
            {
                int neighbor_node_id = graph -> edge[edge_id];

                if(!(articulation_point -> discovery_time[neighbor_node_id]))
                {
                    // Neighbor node hasn't been visited
                        // Push it into Stack ap_stack
                    stackPush(ap_stack, neighbor_node_id);
                        // Update neighbor node's parent as recent node and discovery time
                    articulation_point -> parent[neighbor_node_id] = recent_node;
                    articulation_point -> discovery_time[neighbor_node_id] = articulation_point -> low_time[neighbor_node_id] = ++time;

                    // Exist non visited node such that DFS of the path can keep going, so don't do backward trace
                    backward_trace_flag = false;

                    // trace :: 
                    // cout << "Push " << neighbor_node_id << endl;

                    // Ensure order of item in Stack ap_stack is same as forward list in parallel, cuase we want to track deeply in one way
                    break;
                }

            }

            if(backward_trace_flag)     // Condsidering walk to the end of the path, if so, then do backword trace
            {
                // Pop Stack ap_stack
                stackPop(ap_stack);

                // trace :: 
                // cout << "recent_node = " << recent_node << endl;

                // Record #child of recent_node
                int num_child = 0; 

                // Start to consider the neighbor of recent_node, means the end node of this DFS path, whether it could be discover earlier(exist a backward edge to it's ancestors)
                for(int edge_id = graph -> offset[recent_node] ; edge_id < graph -> offset[recent_node + 1] ; edge_id++)
                {
                    int neighbor_node_id = graph -> edge[edge_id];

                    // recent_node is parent of it's neighbor
                    if(recent_node == articulation_point -> parent[neighbor_node_id])
                    {
                        // Increase #child, cause DFS won't occur "neighbors in same biconnected component got same parent"
                        num_child++;
                        // Update low time of recent_node
                        articulation_point -> low_time[recent_node] = min(articulation_point -> low_time[recent_node], articulation_point -> low_time[neighbor_node_id]);

                        // recent_node is root of the DFS tree and it's #child is greater than 1 or not
                        // if((articulation_point -> parent[recent_node] == -1) && (num_child > 1) && (graph -> degree[recent_node] != 2))
                        if((articulation_point -> parent[recent_node] == -1) && (num_child > 1))
                        {
                            // Ensure recent_node is AP cause "recent_node is root && #child > 1"
                            articulation_point -> ap_flag[recent_node] = 1;
                            articulation_point -> num_ap++;
                            articulation_point -> num_commponent++;

                            // trace :: 
                            // cout << "root node = " << recent_node << " neighbor node = " << neighbor_node_id << endl;
                            
                            break;
                        }

                        // recent_node isn't root of DFS tree and "doesn't exist back edge from my child (my child's lowest discover time must be greater than or equal to mine))"
                        // if((articulation_point -> parent[recent_node] != -1) && (articulation_point -> low_time[neighbor_node_id] >= articulation_point -> discovery_time[articulation_point -> parent[neighbor_node_id]]) && (graph -> degree[recent_node] != 2))
                        if((articulation_point -> parent[recent_node] != -1) && (articulation_point -> low_time[neighbor_node_id] >= articulation_point -> discovery_time[articulation_point -> parent[neighbor_node_id]]))
                        {
                            // Consider recent_node has been consider before or not
                            if(!articulation_point -> ap_flag[recent_node])
                            {
                                articulation_point -> num_ap++;
                            }

                            // set recent_node as AP
                            articulation_point -> ap_flag[recent_node] = 1;
                            articulation_point -> num_commponent++;

                            // trace :: 
                            // cout << "recent node = " << recent_node << " neighbor node = " << neighbor_node_id << endl;
                        }
                    
                    }
                    // Consider recent_node's neighbors that parent is not recent_node, updating recent_node's lowest discover time.
                    else if((neighbor_node_id != articulation_point -> parent[recent_node]) && (articulation_point -> discovery_time[neighbor_node_id] < articulation_point -> discovery_time[recent_node]))
                    {
                        articulation_point -> low_time[recent_node] = min(articulation_point -> low_time[recent_node], articulation_point -> discovery_time[neighbor_node_id]);
                    }                
                }
            }
        }

#ifdef TRACE_PARTITION_GRAPH
    cout << endl;
    cout << "---------- AFTER AP DETECTIONING ----------- " << endl;
    cout << "ARTICULATION POINTS :: " << endl;
    int num_ap = 0;
    for(int node_id = graph_info -> start_node ; node_id <= graph_info -> end_node ; node_id++)
    {
        if(articulation_point -> ap_flag[node_id])
        {
            cout << node_id << "  ";
            num_ap++;
        }
    }
    cout << endl;

    cout << " #AP = " << num_ap << endl;
    cout << " #AP = " << articulation_point -> num_ap << endl;
    cout << " #Compoonent = " << articulation_point -> num_commponent << endl;
#endif /* TRACE_PARTITION_GRAPH */

    }

    // Free the stack;
    stackFree(ap_stack);
}

#pragma endregion /* DETECT_ARTICULATION_POINT */