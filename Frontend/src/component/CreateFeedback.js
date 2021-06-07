import React from "react";
import FeedbackForm from "../component/FeedbackForm";
class CreateFeedback extends React.Component{
    
    componentDidMount () {

    }
    render()
    {
        return(
            <div > 
    <h2 style={{display: 'flex', justifyContent: 'center'}} >Enter feedback information for '{this.props.match.params.testid}' here!</h2>
                    {/* <Button onClick={this.handleClick} style={{justifyContent:'center'}}>Enter a module that you teach!</Button> */}
                    {console.log(this.props.match.params.testid)}
                    <FeedbackForm requestMethod="post"
                        testID={this.props.match.params.testid}
                        userID={this.props.match.params.userid}
                        btnText="Create a feedback!" />
            </div>
        )
    }
}

export default CreateFeedback