import React from "react";
import CustomForm from "../component/Form";
class AddModule extends React.Component{
    render()
    {
        return(
            <div >
                <h2 style={{display: 'flex', justifyContent: 'center'}} >Enter a module Information here!</h2>
                    <CustomForm requestMethod="post"
                        moduleID={null}
                        userID={this.props.match.params.userid}
                        btnText="Create a module!" />
            </div>
        )
    }
}
export default AddModule