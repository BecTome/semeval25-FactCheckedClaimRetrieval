from sklearn.model_selection import train_test_split

def get_split_indices_no_gs_overlap(df, test_size=0.1, random_state=42, stratify_by=None):
    
    dup_criteria = df.duplicated(subset=["full_text"], keep=False)#& (df_posts_train.full_text != "")

    df_dups = df[dup_criteria].copy()
    df_not_dups = df[~dup_criteria].copy()
        
    # Step 1: Flatten the gs column to get all unique gs codes
    all_gs = set([item for sublist in df_not_dups['gs'] for item in sublist])
    # Step 2: Split these unique gs codes into train and test
    gs_train, gs_dev = train_test_split(list(all_gs), test_size=test_size, random_state=random_state)

    gs_dups = set([item for sublist in df_dups['gs'] for item in sublist])
    gs_train = set(gs_train).union(gs_dups)
    gs_train, gs_dev = list(gs_train), list(gs_dev)
    
    
    # Step 3: Create train and test DataFrames by checking if any element in `gs` is in gs_train or gs_test
    mask_in_train = df['gs'].apply(lambda x: any(item in gs_train for item in x))
    mask_in_dev = df['gs'].apply(lambda x: any(item in gs_dev for item in x))
    mask_in_both = mask_in_train & mask_in_dev
    
    idx_train = df[mask_in_train].index
    # idx_train_exclusive = df[mask_in_train&(~mask_in_dev)].index
    # idx_dev = df[mask_in_dev].index
    idx_dev_exclusive = df[mask_in_dev&(~mask_in_train)].index
    
    # idx_both = df[mask_in_both]

    print("Total", len(df))
    print("Train", len(idx_train))
    # print("Train (No Overlap)", len(idx_train_exclusive))
    # print("Dev", len(idx_dev))
    print("Dev (No Overlap)", len(idx_dev_exclusive))
    print("Sum", len(idx_train) + len(idx_dev_exclusive))
    # print("Dev (Intersection)", len(idx_both))

    assert len(df) == len(idx_train) + len(idx_dev_exclusive)

    return idx_train.tolist(), idx_dev_exclusive.tolist()