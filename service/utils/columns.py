class Columns:
    """Fixed column names for tables that contain interactions and recommendations."""

    User = "user_id"
    Item = "item_id"
    TargetItem = "target_item_id"
    Weight = "weight"
    Datetime = "datetime"
    Rank = "rank"
    Score = "score"
    UserItem = [User, Item]
    Interactions = [User, Item, Weight, Datetime]
    Recommendations = [User, Item, Score, Rank]
    RecommendationsI2I = [TargetItem, Item, Score, Rank]
